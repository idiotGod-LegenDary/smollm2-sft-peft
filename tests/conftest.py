"""本地假 Tokenizer：单元测试不依赖网络或 Hugging Face 模型下载。"""
import pytest

class FakeTokenizer:
    pad_token_id = 0
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict=False):
        assert tokenize is True
        assert return_dict is False
        if not messages or messages[0]['role']!='user':
            raise ValueError('需以 user 开头')
        text = '<U>'+messages[0]['content']+'</U><A>'
        if len(messages)==2:
            text += messages[1]['content']+'</A>'
        elif not add_generation_prompt:
            text += '</A>'
        return [ord(c)+10 for c in text]

@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()
