import openai


def generate_recommendation(langchoice_sentence: str, recommended_book, input_query):
    """
    langchoice_sentence : 선택된 언어에 따라 "한국어로 생성해줘" "Please answer in english" 등 생성언어를 지정하는 명령어 str
    recommend_book : 생성할 대상 도서 Bookdata 또는 str
    input_query : 유저입력 str
    return : gpt api complettion json. completion["choices"][0]["message"]["content"]에 실제 생성결과 존재
    """
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a recommendation explainer. "
                    f"You take a user request and one recommended book and explain why they were recommeded in terms of relevance and adequacy. "
                    "You should not make up stuff and explain grounded on provided recommendation data. "
                    f"{langchoice_sentence}"
                    "Keep the explanation short. "
                ),
            },
            {
                "role": "user",
                "content": f"user question:{input_query} recommendations:{recommended_book}",
            },
        ],
    )
    return completion
