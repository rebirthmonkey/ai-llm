#!/usr/bin/env python3

import json

from langchain_openai import ChatOpenAI

def main():
    # 向 LLM 发送一个 prompt
    student_1_description = "David Nguyen is a sophomore majoring in computer science at Stanford University. He is Asian American and has a 3.8 GPA. David is known for his programming skills and is an active member of the university's Robotics Club. He hopes to pursue a career in artificial intelligence after graduating."
    student_2_description = "Ravi Patel is a sophomore majoring in computer science at the University of Michigan. He is South Asian Indian American and has a 3.7 GPA. Ravi is an active member of the university's Chess Club and the South Asian Student Association. He hopes to pursue a career in software engineering after graduating."
    school_1_description = "Stanford University is a private research university located in Stanford, California, United States. It was founded in 1885 by Leland Stanford and his wife, Jane Stanford, in memory of their only child, Leland Stanford Jr. The university is ranked #5 in the world by QS World University Rankings. It has over 17,000 students, including about 7,600 undergraduates and 9,500 graduates23. "
    description = [student_1_description, school_1_description]

    custom_functions = [
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
        },
        {
            'name': 'extract_school_info',
            'description': 'Get the school information from the body of the input text',
            'parameters': {
                'type': 'object',
                'properties': {
                    'name': {
                        'type': 'string',
                        'description': 'Name of the school.'
                    },
                    'ranking': {
                        'type': 'integer',
                        'description': 'QS world ranking of the school.'
                    },
                    'country': {
                        'type': 'string',
                        'description': 'Country of the school.'
                    },
                    'no_of_students': {
                        'type': 'integer',
                        'description': 'Number of students enrolled in the school.'
                    }
                }
            }
        }
    ]

    # 初始化 LLM 模型
    llm = ChatOpenAI(model_name="gpt-4")

    for i in description:
        res = llm.invoke(i, functions=custom_functions, function_call='auto')

        print(res.additional_kwargs.get('function_call').get('arguments'))


if __name__ == "__main__":
    main()
