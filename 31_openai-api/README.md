# OpenAI API开发

## 简介

### Completion



### Chat Completion



### Function Calling

ChatGPT 的 Function Calling 就是给 prompt 出个模板，进行批处理。它允许 ChatGPT 在对话中调用特定的功能来完成更复杂的任务，这些功能可能包括执行计算、检索信息、翻译文本等。但注意，ChatGPT 的 Function Calling 只能让 ChatGPT 进行文字处理，集成外部 API 或内置功能需要根据返回的文件结果进一步出发 API 或函数的调用。其具体步骤包括：

- 解析请求：当用户输入一条消息时，ChatGPT 需要理解这条消息并判断是否需要调用某个函数来回应。这涉及到 NLP 技术来解析用户的意图。
- 确定功能：一旦确定了需要调用的功能，ChatGPT 会根据预设的逻辑或学习到的模式来确定如何执行该功能。
- 执行并返回结果：ChatGPT 根据自身 LLM 逻辑判断调用 Function，并返回文本结果。
- 执行函数：函数跟进返回结果调用相应的 API 或函数，传入必要的参数（如果有的话），并执行。这可能涉及到查询数据库、访问外部 API 或执行内置的算法。
- 返回结果：函数执行完成后，ChatGPT 会接收到结果，并将这个结果整合到回应消息中，返回给用户。

假设 ChatGPT 集成了一个天气查询的 API，用户可以询问当前的天气情况。

- 用户：What's the weather like in Paris today?
- ChatGPT：首先解析这个请求，识别出用户想要查询“今天的巴黎天气”。
- ChatGPT：确定需要调用天气查询的功能，并识别出地点为“巴黎”。
- ChatGPT：向集成的天气 API 发送请求，参数为“巴黎”。
- API 返回结果：API 处理请求并返回巴黎当天的天气数据。
- ChatGPT：将 API 返回的天气数据整合到回应中，例如：“Today in Paris, it's mostly sunny with a high of 18°C and a low of 9°C.”

这个过程展示了 ChatGPT 如何通过函数调用来扩展其回应的能力，提供更具体和有用的信息。

##### 函数结构

它在原先的 ChatGPT API 接口中增加了 2 个与 messages 字段同级别的字段 `funcitons` 和 `function_call`：

- functions：多个 function 的列表，用来告诉 ChatGPT 要从这个列表选择一个 function。function 用来声明一个函数，ChatGPT 选择它以后，会按照它定义的参数格式来返回一个字符串。
  - name：函数名。
  - description：用自然语言描述函数的功能，ChatGPT 是通过它来判断是否要选择这个 function。
  - parameters：返回参数结构，也就是 ChatGPT 返回字符串的格式。
- function_call：控制 ChatGPT 怎么选择 function 的逻辑。
  - auto：自动选
  - none：不选
  - {"name": "xxx"} ：指定选 xxx

```python
student_custom_functions = [
        {
            'name': 'extract_student_info',
            'description': 'Get the student information from the body of the input text',
            'parameters': {
                'type': 'object',
                'properties': {
                    'name': {
                        'type': 'string',
                        'description': 'Name of the person'
                    },
                    'major': {
                        'type': 'string',
                        'description': 'Major subject.'
                    },
                    'school': {
                        'type': 'string',
                        'description': 'The university name.'
                    },
                    'grades': {
                        'type': 'integer',
                        'description': 'GPA of the student.'
                    },
                    'club': {
                        'type': 'string',
                        'description': 'School club for extracuricular activities. '
                    }
                }
            }
        }
    ]
```

## Lab

### Model List

- [OpenAI models API](01_openai.ipynb)

### Completion

- [OpenAI Legacy Completion API](10_openai-completion.ipynb)

### Chat Completion

- [OpenAI Chat Completion API](13_openai-chat-completion.ipynb)
- [OpenAI Tokenization API](14_openai-tiktoken.ipynb)

### Function Calling

- [Function Calling](21_function-call1.ipynb)  
- [Function Calling](22_function-call2.ipynb)  

以下是另一个 Python 程序，当没有 Function Calling 时的代码如下：

```shell
python3 31_standard.py
```

当采用了 Function Calling 后，代码如下：

```shell
python3 32_prompt-process.py
```

最后，让 ChatGPT 通过 prompt 的内容自动选择 funciton list 中的某个 function 调用：

```shell
python3 33_prompt-multi-process.py
```

### Translator

- [Translator](37_translator/README.md)：基于OpenAI API的Translator应用开发
