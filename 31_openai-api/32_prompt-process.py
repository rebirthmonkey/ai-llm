#!/usr/bin/env python3

import json

from langchain_openai import ChatOpenAI

def main():
    # 向 LLM 发送一个 prompt
    student_1_description = "David Nguyen is a sophomore majoring in computer science at Stanford University. He is Asian American and has a 3.8 GPA. David is known for his programming skills and is an active member of the university's Robotics Club. He hopes to pursue a career in artificial intelligence after graduating."
    student_2_description = "Ravi Patel is a sophomore majoring in computer science at the University of Michigan. He is South Asian Indian American and has a 3.7 GPA. Ravi is an active member of the university's Chess Club and the South Asian Student Association. He hopes to pursue a career in software engineering after graduating."
    student_description = [student_1_description, student_2_description]

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
                        'description': 'School club for extracurricular activities. '
                    }
                }
            }
        }
    ]

    # 初始化 LLM 模型
    llm = ChatOpenAI(model_name="gpt-4")

    for i in student_description:
        res = llm.invoke(i, functions=student_custom_functions, function_call='auto')

        print(res.to_json().get('kwargs').get('content'))


if __name__ == "__main__":
    main()
