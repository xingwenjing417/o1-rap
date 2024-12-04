from abc import ABC, abstractmethod

import torch

#from llama import LLaMA


class QueryLM(ABC):
    @abstractmethod
    def query_LM(self, prompt, **gen_kwargs):
        pass

    @abstractmethod
    def query_next_token(self, prompt: list[str]):
        pass


class QueryHfModel(QueryLM):
    # This is not well-tested. Please use LLaMA if possible.
    def query_next_token(self, prompt: list[str]):
        raise NotImplementedError

    def __init__(self, model, tokenizer, max_response_length, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_examples = 1
        self.max_response_length = max_response_length

    def query_LM(self, prompt, **gen_kwargs):
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            # print("input length", len(inputs))
            # Generate
            generate_ids = self.model.generate(inputs.input_ids, max_new_tokens=self.max_response_length, **gen_kwargs)
            text = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return text


class QueryLlama(QueryLM):
    def __init__(self, llamamodel: LLaMA, max_response_length, log_file) -> None:
        self.llamamodel = llamamodel
        self.tokenizer = self.llamamodel.tokenizer
        self.max_response_length = max_response_length
        self.log_file = log_file
        self.max_batch_size = llamamodel.model.params.max_batch_size
        self.yes_no = self.tokenizer.encode('Yes No', bos=False, eos=False)#self.yes_no 变量将保存 "Yes No" 在 LLaMA 模型中的 token 编码。

    #基于当前批量的提示文本（通过将prompt重复end - start次得到）进行文本生成操作。在生成过程中，设置了最大生成长度为self.max_response_length，温度参数为temperature，以及结束标记 ID 为eos_token_id
    def query_LM(self, prompt, eos_token_id, num_return_sequences=1, do_sample=True, temperature=0.8):
        temperature = temperature if do_sample else 0
        all_results = []
        for start in range(0, num_return_sequences, self.max_batch_size):
            end = min(start + self.max_batch_size, num_return_sequences)
            results = self.llamamodel.generate([prompt] * (end - start), max_gen_len=self.max_response_length, temperature=temperature, eos_token_id=eos_token_id)
            all_results.extend(results)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write("="*50+"\n")
                f.write(prompt + "\n")
                for result in all_results:
                    f.write("-"*50+"\n")
                    f.write(result.replace(prompt, "") + "\n")
        return all_results

    @torch.no_grad()
    def query_next_token(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        ret = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
            tokens = torch.tensor([tokens]).cuda().long()
            output, h = self.llamamodel.model.forward(tokens, start_pos=0)
            ret.append(output)
        outputs = torch.cat(ret, dim=0)
        #从 outputs 中提取出与 "Yes No" token 对应的列， 是为了计算模型是否倾向于生成 "Yes" 或 "No"。通过对这些 logits 进行处理，模型可以根据其对 "Yes" 和 "No" 的信心水平来做出决策。这对于某些任务（如二分类任务，或者判断模型是否给出了“是”或“否”的答案）非常重要。
        filtered = outputs[:, self.yes_no]
        dist = torch.softmax(filtered, dim=-1)
        return dist

