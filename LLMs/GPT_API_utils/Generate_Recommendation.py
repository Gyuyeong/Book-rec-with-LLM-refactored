import openai


def generate_recommendation(langchoice_sentence: str, recommended_book, input_query):
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
