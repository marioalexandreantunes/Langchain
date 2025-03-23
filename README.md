# Projeto Langchain

## Configuração do Ambiente

Este projeto utiliza variáveis de ambiente para chaves de API e outras informações sensíveis. Estas são carregadas a partir de um ficheiro `.env` utilizando python-dotenv.

### Instruções de Configuração

1. Instale as dependências necessárias:
   ```
   pip install -r requirements.txt
   ```

2. Configure as suas variáveis de ambiente:
   - Renomeie o ficheiro `.env` se tiver um nome diferente
   - Adicione as suas chaves de API ao ficheiro `.env`:
     ```
     GROQ_API_KEY=sua_chave_api_groq_aqui
     APITUBE_KEY=sua_chave_apitube_aqui
     ```

3. Execute a aplicação:
   ```
   python main.py
   ```

## Funcionalidades

- Utiliza Langchain com o modelo Groq LLM para processamento de linguagem natural
- Inclui ferramentas para operações matemáticas (soma e subtração)
- Fornece funcionalidade de pesquisa de notícias sobre criptomoedas
- Integra a pesquisa DuckDuckGo para obtenção de informações gerais

## Sobre o main.py

O ficheiro `main.py` serve como um exemplo prático de implementação do Langchain. Demonstra como:

- Configurar um agente Langchain com o modelo Groq LLM
- Definir ferramentas personalizadas (operações matemáticas e pesquisa de notícias)
- Integrar ferramentas externas como o DuckDuckGo
- Criar um sistema de processamento de linguagem natural com capacidade de utilizar ferramentas adequadas para responder a perguntas

Este exemplo pode ser utilizado como base para desenvolver aplicações mais complexas utilizando o framework Langchain.
