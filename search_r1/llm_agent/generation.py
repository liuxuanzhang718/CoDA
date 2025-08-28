import time
import logging
from pydantic.main import TupleGenerator
import torch
import re
from collections import defaultdict

import random
from typing import List, Dict, Any, Tuple, Union
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
import numpy as np

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool = False
    search_url: str = ""
    topk: int = 3
    # Hierarchical tool calling parameters
    enable_hierarchical: bool = True
    inner_max_turns: int = 5
    inner_system_prompt: str = """You are a helpful assistant excel at answering questions with multi-turn search engine calling.
To answer questions, you must first reason through the available information using <think> and </think>.
If you identify missing knowledge, you may issue a search request using <search> query </search> at any time.
The retrieval system will provide you with the most relevant documents enclosed in <documents> and </documents>.
After each search, you need to summarize and refine the existing documents in <refine> and </refine>.
You may send multiple search requests if needed.
Once you have sufficient information, provide a concise final answer using <answer> and </answer>. For example, <answer> Donald Trump </answer>.

Question: """
    inner_system_prompt_instruct: str = """You are a helpful assistant excel at answering questions with multi-turn search engine calling.
To answer questions, you must first reason through the available information using <think> and </think>.
If you identify missing knowledge, you may issue a search request using <search> query </search> at any time.
The retrieval system will provide you with the most relevant documents enclosed in <documents> and </documents>.
After each search, you need to summarize and refine the existing documents in <refine> and </refine>.
You may send multiple search requests if needed.
Once you have sufficient information, provide a concise final answer using <answer> and </answer>. For example, <answer> Donald Trump </answer>."""

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
        model_type: str = "base"
    ) -> None:
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.model_type = model_type

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor, is_outer_loop: bool = True) -> Tuple[torch.Tensor, List[str]]:
        """Process responses to stop at the first complete action operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        # Define which actions to look for based on loop level
        if is_outer_loop and self.config.enable_hierarchical:
            # Outer loop: look for task and answer actions
            action_tags = ['</task>', '</answer>']
        else:
            # Inner loop or non-hierarchical: look for search and answer actions
            action_tags = ['</search>', '</answer>']

        processed_responses = []
        for resp in responses_str:
            # Find the first occurrence of any action tag
            earliest_pos = len(resp)
            chosen_tag = None
            
            for tag in action_tags:
                pos = resp.find(tag)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
                    chosen_tag = tag
            
            if chosen_tag:
                # Truncate at the end of the first action tag found
                processed_responses.append(resp.split(chosen_tag)[0] + chosen_tag)
            else:
                processed_responses.append(resp)
        
        responses_str = processed_responses

        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            logging.warning(f"Observation too long: {next_obs_ids.shape[1]} > {self.config.max_obs_length}. Consider increasing max_obs_length in config.")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> DataProto:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            new_response_ids, new_response_info_mask = self._info_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['response_info_mask'],
                cur_responses,
                next_obs_ids, 
                pad_to_left=False
            )
        else:
            new_response_ids, new_response_info_mask = self._info_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['response_info_mask'],
                cur_responses,
                pad_to_left=False
            )
        effective_len = self.tensor_fn.create_attention_mask(new_response_ids).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': new_response_ids[:, :max_len], 'response_info_mask': new_response_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> Tuple[DataProto, float, torch.Tensor]:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
            Returns a tuple of (output_dataproto, generation_time, generated_tokens_per_sample)
        """
        num_gpus = self.config.num_gpus
        
        start_time = time.time()  # Record generation start time

        if num_gpus <= 1:
            output = self.actor_rollout_wg.generate_sequences(active_batch)
            padding_size = 0
        else:
            batch_size = active_batch.batch['input_ids'].shape[0]
            remainder = batch_size % num_gpus
            
            for key in active_batch.batch.keys():
                active_batch.batch[key] = active_batch.batch[key].long()
            if remainder == 0:
                output = self.actor_rollout_wg.generate_sequences(active_batch)
                padding_size = 0
            else:
                # Add padding sequences
                padding_size = num_gpus - remainder
                padded_batch = {}
                
                for k, v in active_batch.batch.items():
                    # Use first sequence as padding template
                    pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
                    padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

                padded_active_batch = DataProto.from_dict(padded_batch)
                for key in padded_active_batch.batch.keys():
                    padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

                # Generate with padded batch
                output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        gen_time = time.time() - start_time  # Calculate generation time

        # Calculate generated tokens
        responses = output.batch['responses']
        generated_tokens = (responses != self.tokenizer.pad_token_id).sum(dim=1)  # Count generated tokens per sample

        # Remove padding from output
        if padding_size > 0:
            trimmed_batch = {k: v[:-padding_size] for k, v in output.batch.items()}
            
            # Handle meta_info if present
            if hasattr(output, 'meta_info') and output.meta_info:
                trimmed_meta = {}
                for k, v in output.meta_info.items():
                    if isinstance(v, torch.Tensor):
                        trimmed_meta[k] = v[:-padding_size]
                    else:
                        trimmed_meta[k] = v
                output.meta_info = trimmed_meta
                
            output.batch = trimmed_batch
            generated_tokens = generated_tokens[:-padding_size]  # Remove padding from token counts
        
        return output, gen_time, generated_tokens

    def run_llm_loop(self, gen_batch: DataProto, initial_input_ids: torch.Tensor) -> DataProto:
        """Run main LLM generation loop."""

        start_time = time.time()
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        right_side = {'responses': initial_input_ids[:, []], 'response_info_mask': initial_input_ids[:, []]}

        batch_size = gen_batch.batch['input_ids'].shape[0]
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        executor_data = []

        executor_calls_count = torch.zeros(batch_size, dtype=torch.int)
        completion_times = torch.zeros(batch_size, dtype=torch.float32)  # Track completion time for each sample
        total_gen_tokens = torch.zeros(batch_size, dtype=torch.float32)
        total_gen_time = torch.zeros(batch_size, dtype=torch.float32)
        context_lengths = [[] for _ in range(batch_size)]

        # Main generation loop
        for _ in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            

            active_indices = torch.where(active_mask)[0]
            current_lengths = (rollings_active.batch['input_ids'] != self.tokenizer.pad_token_id).sum(dim=1).cpu().tolist()
            for idx, length in zip(active_indices, current_lengths):
                context_lengths[idx].append(length)

            gen_output, gen_time, gen_tokens = self._generate_with_gpu_padding(rollings_active)
            total_gen_tokens[active_mask] += gen_tokens.cpu()
            total_gen_time[active_mask] += gen_time

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'], is_outer_loop=True)
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)


            
            # Execute in environment and process observations
            next_obs, dones, inner_trajs, task_counts = self.execute_predictions(
                responses_str, active_mask, is_inner_loop=False
            )
            if task_counts is not None:
                # We need to make sure we only add counts for active samples.
                # execute_predictions returns counts for all original predictions passed.
                executor_calls_count += task_counts * active_mask.int().to(task_counts.device)



            if isinstance(inner_trajs, DataProto):
                for key in gen_batch.non_tensor_batch.keys():
                    origin_idx = inner_trajs.non_tensor_batch['task_origin']
                    inner_trajs.non_tensor_batch[key] = gen_batch.non_tensor_batch[key][origin_idx]

                inner_trajs.non_tensor_batch.pop('task_origin')
                executor_data.append(inner_trajs)
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)

            just_finished_mask = active_mask & (~curr_active_mask)
            if just_finished_mask.any():
                elapsed_time = time.time() - start_time
                # Only set the time if it hasn't been set before
                completion_times[just_finished_mask] = elapsed_time

            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            right_side = self._update_right_side(
                right_side,
                responses_ids,
                next_obs_ids
            )

        if active_mask.any():
            total_elapsed_time = time.time() - start_time
            completion_times[active_mask] = total_elapsed_time
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()

        logging.info(f"Active trajectory count per turn: {active_num_list}")
        
        data = self._compose_final_output(original_left_side, right_side, meta_info)
        # Add planner-specific metrics to non_tensor_batch
        data.non_tensor_batch['sample_time_s'] = completion_times.tolist()  # Add sample completion times to metadata
        data.non_tensor_batch['executor_calls_count'] = executor_calls_count.tolist()
        data.non_tensor_batch['executor_loop_count'] = [0] * batch_size
        avg_gen_speed = torch.nan_to_num(total_gen_tokens / total_gen_time, nan=0.0).tolist()
        data.non_tensor_batch['avg_gen_speed'] = avg_gen_speed
        avg_context_length = [np.mean(lengths) if lengths else 0 for lengths in context_lengths]
        data.non_tensor_batch['avg_context_length'] = avg_context_length

        for key in gen_batch.non_tensor_batch.keys():
            data.non_tensor_batch[key] = gen_batch.non_tensor_batch[key]
        
        # Align and concatenate trajectories
        data = self._align_and_concat_trajectories(data, executor_data)

        # Post-processing: ensure batch size is divisible by num_gpus
        if self.config.num_gpus > 1:
            current_batch_size = len(data)
            remainder = current_batch_size % self.config.num_gpus
            
            if remainder != 0:
                padding_size = self.config.num_gpus - remainder
                logging.info(f"Padding {padding_size} samples to make batch size divisible by {self.config.num_gpus} GPUs")
                
                # Randomly select indices for padding
                padding_indices = random.choices(range(current_batch_size), k=padding_size)
                
                # Create padding samples using select_idxs
                padding_samples = data.select_idxs(padding_indices)
                
                # Concatenate original data with padding samples
                data = DataProto.concat([data, padding_samples])

        assert len(data) % self.config.num_gpus == 0, "Batch size after generation must be divisible by num_gpus"
        
        return data

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict = {}) -> DataProto:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        prompt_tokens = (final_output['prompts'] != self.tokenizer.pad_token_id).sum(dim=1).cpu().tolist()
        response_tokens = (final_output['responses'] != self.tokenizer.pad_token_id).sum(dim=1).cpu().tolist()

        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['response_info_mask'])
        ], dim=1)

        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)

        final_output.non_tensor_batch['prompt_tokens'] = prompt_tokens
        final_output.non_tensor_batch['response_tokens'] = response_tokens
        
        return final_output

    def _align_and_concat_trajectories(self, data: DataProto, executor_data: List[DataProto]) -> DataProto:
        """Align and concatenate trajectories with left-padded prompts and right-padded responses."""
        if not executor_data:
            return data
            
        all_trajs = [data] + executor_data
        pad_token_id = self.tokenizer.pad_token_id
        
        # Find max lengths
        max_prompt_len = max(traj.batch['prompts'].shape[1] for traj in all_trajs)
        max_response_len = max(traj.batch['responses'].shape[1] for traj in all_trajs)
        
        def pad_tensor(tensor, target_len, left_pad=True):
            if tensor.shape[1] >= target_len:
                return tensor
            pad_len = target_len - tensor.shape[1]
            padding = torch.full((tensor.shape[0], pad_len), pad_token_id, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([padding, tensor] if left_pad else [tensor, padding], dim=1)
        
        def align_traj(traj):
            prompts = pad_tensor(traj.batch['prompts'], max_prompt_len, left_pad=True)
            response_ids = pad_tensor(traj.batch['responses'], max_response_len, left_pad=False) 
            response_info_mask = pad_tensor(traj.batch['response_info_mask'], max_response_len, left_pad=False)
            
            input_ids = torch.cat([prompts, response_ids], dim=1)
            attention_mask = self.tensor_fn.create_attention_mask(input_ids)
            info_mask = torch.cat([
                self.tensor_fn.create_attention_mask(prompts),
                self.tensor_fn.create_attention_mask(response_info_mask)
            ], dim=1)
            
            aligned_batch = {
                'prompts': prompts, 'responses': response_ids, 'response_info_mask': response_info_mask,
                'input_ids': input_ids, 'attention_mask': attention_mask, 'info_mask': info_mask,
                'position_ids': self.tensor_fn.create_position_ids(attention_mask)
            }
            
            aligned_traj = DataProto.from_dict(aligned_batch)
            aligned_traj.meta_info.update(getattr(traj, 'meta_info', {}))
            if hasattr(traj, 'non_tensor_batch'):
                aligned_traj.non_tensor_batch = traj.non_tensor_batch
            return aligned_traj
        
        result = align_traj(all_trajs[0])
        for traj in all_trajs[1:]:
            result = DataProto.concat([result, align_traj(traj)])
        
        return result

    def execute_predictions(self, predictions: List[str], active_mask: torch.Tensor, is_inner_loop: bool = False) -> Tuple[List[str], List[int], Union[None, DataProto], torch.Tensor]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            predictions: List of action predictions
            pad_token: Token to use for padding
            active_mask: Mask indicating which sequences are active
            is_inner_loop: Whether this is being called from inner loop
            
        Returns:
            Tuple of (next_obs, dones)
        """
        cur_actions, contents = self.postprocess_predictions(predictions, is_outer_loop=not is_inner_loop)

        if not is_inner_loop:
            task_counts = torch.tensor([1 if action == 'task' else 0 for action in cur_actions], dtype=torch.int)
        else:
            # No tasks in inner loop
            task_counts = torch.zeros(len(cur_actions), dtype=torch.int)
        next_obs, dones = [], []
        has_task = False
        
        if is_inner_loop:
            search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            task_queries = [content for action, content in zip(cur_actions, contents) if action == 'task']
            task_origin = np.array([idx for idx, action in enumerate(cur_actions) if action == 'task'])
            if len(task_queries):
                has_task = True
                executor_trajs, task_results = self.batch_execute(task_queries)
                executor_trajs.non_tensor_batch['task_origin'] = task_origin
                assert len(task_results) == sum([1 for action in cur_actions if action == 'task'])

        for (action, active) in zip(cur_actions, active_mask):
            
            if not active:
                next_obs.append('')
                dones.append(1)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    continue

                if is_inner_loop:
                    if action == 'search':
                        next_obs.append(f'\n<documents>{search_results.pop(0).strip()}</documents>\n')
                        dones.append(0)
                    else:
                        error_msg = f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n'
                        next_obs.append(error_msg)
                        dones.append(0)
                else:
                    if action == 'task':
                        next_obs.append(f'\n<result>{task_results.pop(0).strip()}</result>\n')
                        dones.append(0)
                    else:
                        error_msg = f'\nMy previous action is invalid. \
If I want to assign a task, I should put the task content between <task> and </task>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n'
                        next_obs.append(error_msg)
                        dones.append(0)

        if has_task:
            return next_obs, dones, executor_trajs, task_counts  # Return with executor trajectories
        else:
            return next_obs, dones, None, task_counts

    def postprocess_predictions(self, predictions: List[Any], is_outer_loop: bool = True) -> Tuple[List[str], List[str]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            is_outer_loop: Whether this is the outer loop (supports task action) or inner loop (only search/answer)
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                if is_outer_loop and self.config.enable_hierarchical:
                    # Outer loop: support task and answer actions
                    pattern = r'<(task|answer)>(.*?)</\1>'
                else:
                    # Inner loop or non-hierarchical: support search and answer actions
                    pattern = r'<(search|answer)>(.*?)</\1>'
                    
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_execute(self, queries: List[str]) -> Tuple[DataProto, List[str]]:
        # print(f"executor循环-batch task内容: {queries}")
        
        # Create inner context
        prompts = []
        
        for query in queries:
            if self.model_type == "base":
                inner_prompt = self.config.inner_system_prompt + query + "\n" + "Assistant: "
            else:
                chat_list = [
                    {"role": "system", "content": self.config.inner_system_prompt_instruct},
                    {"role": "user", "content": query}
                ]
                inner_prompt = self.tokenizer.apply_chat_template(chat_list, tokenize=False, add_generation_prompt=True)

            prompts.append(inner_prompt)
        
        # Tokenize inner prompt
        padding_bak = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        inner_prompt_ids = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding='longest',
            add_special_tokens=False
        )['input_ids']


        self.tokenizer.padding_side = padding_bak

        batch_size = len(prompts)
        task_results = [''] * batch_size
        
        # Initialize inner loop state
        inner_gen_batch = DataProto.from_dict({
            'input_ids': inner_prompt_ids,
            'attention_mask': self.tensor_fn.create_attention_mask(inner_prompt_ids),
            'position_ids': self.tensor_fn.create_position_ids(self.tensor_fn.create_attention_mask(inner_prompt_ids))
        })

        original_left_side = {'input_ids': inner_prompt_ids[:, -self.config.max_start_length:]}
        right_side = {'responses': inner_prompt_ids[:, []], 'response_info_mask': inner_prompt_ids[:, []]}
        
        inner_active_mask = torch.ones(batch_size, dtype=torch.bool)
        inner_rollings = inner_gen_batch
        
        inner_loop_count = torch.zeros(batch_size, dtype=torch.int)
        total_gen_tokens = torch.zeros(batch_size, dtype=torch.float32)
        total_gen_time = torch.zeros(batch_size, dtype=torch.float32)
        context_lengths = [[] for _ in range(batch_size)]

        # Inner generation loop
        for _ in range(self.config.inner_max_turns):

            if not inner_active_mask.sum():
                break

            inner_loop_count[inner_active_mask] += 1

            inner_rollings.batch = self.tensor_fn.cut_to_effective_len(
                inner_rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # Generate response in inner loop
            inner_rollings_active = DataProto.from_dict({
                k: v[inner_active_mask] for k, v in inner_rollings.batch.items()
            })

            # Record context length for active samples
            active_indices = torch.where(inner_active_mask)[0]
            current_lengths = (inner_rollings_active.batch['input_ids'] != self.tokenizer.pad_token_id).sum(dim=1).cpu().tolist()
            for idx, length in zip(active_indices, current_lengths):
                context_lengths[idx].append(length)

            inner_gen_output, gen_time, gen_tokens = self._generate_with_gpu_padding(inner_rollings_active)
            total_gen_tokens[inner_active_mask] += gen_tokens.cpu()
            total_gen_time[inner_active_mask] += gen_time

            
            inner_responses_ids, inner_responses_str = self._postprocess_responses(inner_gen_output.batch['responses'], is_outer_loop=False)
            inner_responses_ids, inner_responses_str = self.tensor_fn._example_level_pad(
                inner_responses_ids, inner_responses_str, inner_active_mask
            )
            

            cur_actions, contents = self.postprocess_predictions(inner_responses_str, is_outer_loop=False)
            for idx, (action, content) in enumerate(zip(cur_actions, contents)):
                if action == 'answer':
                    task_results[idx] = content
            
            # Execute predictions in inner loop (search/answer only)
            inner_next_obs, inner_dones, _, _ = self.execute_predictions(
                inner_responses_str, inner_active_mask, is_inner_loop=True
            )
            
            inner_curr_active_mask = torch.tensor([not done for done in inner_dones], dtype=torch.bool)
            inner_active_mask = inner_active_mask * inner_curr_active_mask
            
            inner_next_obs_ids = self._process_next_obs(inner_next_obs)
            
            # Update inner rolling state
            inner_rollings = self._update_rolling_state(
                inner_rollings,
                inner_responses_ids,
                inner_next_obs_ids
            )

            right_side = self._update_right_side(
                right_side,
                inner_responses_ids,
                inner_next_obs_ids
            )
            
        final_output = self._compose_final_output(original_left_side, right_side)
        # Add placeholder for planner-specific metric to ensure key consistency
        final_output.non_tensor_batch['sample_time_s'] = [0] * batch_size  # Placeholder for sample timing in executor
        final_output.non_tensor_batch['executor_calls_count'] = [0] * batch_size
        final_output.non_tensor_batch['executor_loop_count'] = inner_loop_count.tolist()
        avg_gen_speed = torch.nan_to_num(total_gen_tokens / total_gen_time, nan=0.0).tolist()
        final_output.non_tensor_batch['avg_gen_speed'] = avg_gen_speed
        avg_context_length = [np.mean(lengths) if lengths else 0 for lengths in context_lengths]
        final_output.non_tensor_batch['avg_context_length'] = avg_context_length
        

        return final_output, task_results
    
    def batch_search(self, queries: List[str]) -> List[str]:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        try:
            search_response = self._batch_search(queries)
            if 'result' not in search_response:
                raise ValueError("Invalid search response format: missing 'result' key")
            results = search_response['result']
            return [self._passages2string(result) for result in results]
        except (KeyError, ValueError) as e:
            raise ValueError(f"Failed to process search results: {str(e)}")

    def _batch_search(self, queries: List[str]) -> Dict[str, Any]:
        """
        Perform batch search with error handling.

        Args:
            queries: List of search queries

        Returns:
            Search response dictionary

        Raises:
            ValueError: If search URL is not configured
            requests.RequestException: If network request fails
        """
        if not self.config.search_url:
            raise ValueError("Search URL is not configured")

        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }

        try:
            response = requests.post(self.config.search_url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise requests.RequestException(f"Search request failed: {str(e)}")

    def _passages2string(self, retrieval_result: List[Dict[str, Any]]) -> str:
        """
        Convert retrieval results to formatted string.

        Args:
            retrieval_result: List of retrieved documents

        Returns:
            Formatted string representation of search results
        """
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            try:
                content = doc_item['document']['contents']
                content_lines = content.split("\n")
                title = content_lines[0] if content_lines else "Untitled"
                text = "\n".join(content_lines[1:]) if len(content_lines) > 1 else ""
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            except (KeyError, IndexError) as e:
                # Skip malformed documents and continue
                continue

        return format_reference
