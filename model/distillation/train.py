from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime

class Studen_Model():
    def __init__(self):
        pass

class Distill_Model:
    def __init__(self, teacher_config_path, student_config_path):
        self.teacher_config_path = teacher_config_path
        self.student_config_path = student_config_path
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.teacher_tokenizer = self._load_tokenizer(self.teacher_config_path)
        self.student_tokenizer = self._load_tokenizer(self.student_config_path)
    
    def init_weights(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def load_model(self, config_path, random_weight=False):
        """
        load 模型
        """
        
        if random_weight:
            model_config = AutoConfig.from_pretrained(config_path)
            model = AutoModelForCausalLM.from_config(model_config).to(self.device)
            model.apply(self.init_weights)
        else:
            model = AutoModelForCausalLM.from_pretrained(config_path, trust_remote_code=True).to(self.device)
        return model
    
    def _load_tokenizer(self, config_path):
        return AutoTokenizer.from_pretrained(config_path)

    def prepare_data(self, data:str, tokenizer):
        """
        一条字符串，然后可以进行tokenizer化，返回一个向量pt。
        """
        prompt = data
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        print(text)
        model_inputs = tokenizer([text], return_tensors="pt").to(self.device)
        return model_inputs

    def decode_data(self, outputs:torch.Tensor, tokenizer, print_output=False):
        """
        解码输出
        """
        output_ids = outputs.logits.argmax(dim=-1)[0] 
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        if print_output:
            print("thinking content:", thinking_content)
            print("content:", content)
        return thinking_content, content

    def get_teacher_model_output(self, model, tokenizer, input_str:str):
        """
        获取 teacher 模型的输出
        """
        model_inputs = self.prepare_data(data=input_str, tokenizer=tokenizer)
        config = {"output_hidden_states":True, "output_attentions":True}
        with torch.no_grad():
            outputs = model(model_inputs["input_ids"], **config)
        return outputs

    def align_hidden_output(self):
        pass

    def prepare_student_labels(self, test_data=None):
        """
        准备 student 的语料
            test_data 用于测试
        """
        teacher_model = self.load_model(config_path=self.teacher_config_path)
        if test_data is not None:
            outputs = self.get_teacher_model_output(model=teacher_model, 
                                                    tokenizer=self.teacher_tokenizer, 
                                                    input_str=test_data)
            print(outputs)

class Train_model:
    def __init__(self):
        pass
    
    def run(self):
        """
        begin_train
        """
        pass
    
    
        
    
if __name__=="__main__":
    # 如果有本地的config文件
    config_path = "./model/qwen3-0_6B/"  # 模型位置

    data = "Give me a short introduction to large language model."

    distill_model_handler = Distill_Model(
        teacher_config_path=config_path, 
        student_config_path=config_path,
    )
    
    outputs = distill_model_handler.prepare_student_labels(test_data=data)
    print(outputs)