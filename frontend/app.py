from typing import TypedDict

import gradio as gr


class Data(TypedDict):
    name: str
    age: int
    gender: str
    favorite_color: str
    favorite_pet: str
    favorite_food: str
    paragraph: str


# save data to db
def save_data(
    name: str, age: int, gender: str, color: str, pet: str, food: str, paragraph: str
) -> str:
    return str(
        Data(
            name=name,
            age=age,
            gender=gender,
            favorite_color=color,
            favorite_pet=pet,
            favorite_food=food,
            paragraph=paragraph,
        )
    )


demo = gr.Interface(
    fn=save_data,
    inputs=["text", "text", "text", "text", "text", "text", "text"],
    outputs=["text"],
)

demo.launch()
