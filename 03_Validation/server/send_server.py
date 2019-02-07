import requests
import pandas as pd


def send_2_server(y_pred):
    # Enviando previsões realizadas com o modelo para o servidor
    URL = "https://aydanomachado.com/mlclass/03_Validation.php"

    # TODO Substituir pela sua chave aqui
    DEV_KEY = "Dual Core"

    # json para ser enviado para o servidor
    data = {'dev_key': DEV_KEY,
            'predictions': pd.Series(y_pred).to_json(orient='values')}

    # Enviando requisição e salvando o objeto resposta
    r = requests.post(url=URL, data=data)

    # Extraindo e imprimindo o texto da resposta
    pastebin_url = r.text
    print(" - Resposta do servidor:\n", r.text, "\n")
