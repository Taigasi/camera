import whisper
import os
import datetime

# 1. Configurações de Caminho (O mesmo que já funcionou)
pasta_script = os.path.dirname(os.path.abspath(__file__))
nome_arquivo = "audio.ogg"  # Atualize para o nome do seu arquivo atual
caminho_completo = os.path.join(pasta_script, nome_arquivo)

# 2. Verifica se o arquivo existe
if not os.path.exists(caminho_completo):
    print(f"ERRO: Arquivo '{nome_arquivo}' não encontrado na pasta.")
else:
    try:
        print("--> Carregando modelo e iniciando transcrição...")
        modelo = whisper.load_model("base")
        resultado = modelo.transcribe(caminho_completo)
        texto_transcrito = resultado['text'].strip()

        # 3. Formatação estilo "ABNT" (Cabeçalho + Texto)
        data_hoje = datetime.datetime.now().strftime("%d/%m/%Y")
        
        # Cria o conteúdo do arquivo
        conteudo_arquivo = f"""
TÍTULO: Transcrição do arquivo {nome_arquivo}

----------------------------------------------------------------------

{texto_transcrito}
"""
        
        # 4. Salva no arquivo .txt com codificação UTF-8 (Corrige os acentos)
        nome_saida = "transcricao_abnt.txt"
        caminho_saida = os.path.join(pasta_script, nome_saida)

        with open(caminho_saida, "w", encoding="utf-8") as arquivo:
            arquivo.write(conteudo_arquivo.strip())

        print(f"\n=== SUCESSO! ===")
        print(f"Arquivo salvo em: {caminho_saida}")
        print("Abra o arquivo de texto para ver os acentos corretos.")

    except Exception as e:
        print(f"Ocorreu um erro: {e}")
