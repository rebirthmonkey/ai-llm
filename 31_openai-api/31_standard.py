#!/usr/bin/env python3

import json

from langchain_openai import ChatOpenAI


def main():
    # 初始化 LLM 模型
    llm = ChatOpenAI(model_name="gpt-4")

    # 向 LLM 发送一个 prompt
    student_1_description = "David Nguyen is a sophomore majoring in computer science at Stanford University. He is Asian American and has a 3.8 GPA. David is known for his programming skills and is an active member of the university's Robotics Club. He hopes to pursue a career in artificial intelligence after graduating."

    # A simple prompt to extract information from "student_description" in a JSON format.
    prompt1 = f'''
    Please extract the following information from the given text and return it as a JSON object:

    name
    major
    school
    grades
    club

    This is the body of text to extract the information from:
    {student_1_description}
    '''

    # 使用回复
    json_res1 = llm.invoke(prompt1).to_json().get('kwargs').get('content')


    student_2_description = "Ravi Patel is a sophomore majoring in computer science at the University of Michigan. He is South Asian Indian American and has a 3.7 GPA. Ravi is an active member of the university's Chess Club and the South Asian Student Association. He hopes to pursue a career in software engineering after graduating."
    prompt2 = f'''
    Please extract the following information from the given text and return it as a JSON object:

    name
    major
    school
    grades
    club

    This is the body of text to extract the information from:
    {student_2_description}
    '''

    json_res2 = llm.invoke(prompt2).to_json().get('kwargs').get('content')

    # 打印出回复
    print(json_res2)


if __name__ == "__main__":
    main()
