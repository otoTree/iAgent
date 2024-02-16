import connectLLM.wenxin as wx

chat = wx.CHAT()

import flet as ft


def card(role, content):
    if role == 'user':
        icon = 'ft.icons.BEACH_ACCESS'
    else:
        icon = 'ft.icons.ALBUM'
    c = ft.Card(
        content=ft.Container(
            content=ft.Column(
                [
                    ft.ListTile(

                        leading=ft.Icon(eval(icon)),
                        title=ft.Text(role),
                        subtitle=ft.Text(content),
                    ),

                ],

            ),
            width=200,

            padding=10,
        )
    )

    return c


def main(page: ft.Page):
    lv = ft.ListView(expand=1, spacing=10, padding=20, auto_scroll=True, )

    new_message = ft.TextField(hint_text="Please enter text here")
    new_message.border_color = '#3C3C3C'

    def send_click(e):
        lv.controls.append(card('user', new_message.value))

        page.update()

        chat.chat(new_message.value)

        print(chat.message)

        lv.controls.append(card(chat.message[-1]['role'], chat.message[-1]['content']))
        chat.message = []
        new_message.value = ""
        page.update()

    page.add(
        lv,
        ft.Row(controls=[new_message, ft.ElevatedButton("Send", on_click=send_click)],
               alignment=ft.MainAxisAlignment.CENTER, )
    )


ft.app(target=main)
