import json
import re
import os
import time
import rich.console as console
import datetime
import requests
from dotenv import load_dotenv

# Carregar variáveis de ambiente do ficheiro .env
load_dotenv()

# Importar componentes necessários do Langchain
from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from rich.panel import Panel
from langchain_community.tools import DuckDuckGoSearchRun
from typing import List

# Inicializar o console para apresentação formatada
console = console.Console()


class CustomOutputParser(AgentOutputParser):
    """Analisador personalizado para processar a saída do modelo de linguagem."""
    def parse(self, llm_output: str):
        # Verificar se é uma resposta final
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        # Extrair ação e entrada da ação usando expressões regulares
        action_match = re.search(r"Action: (.*?)[\n]", llm_output)
        action_input_match = re.search(r"Action Input: (.*)", llm_output)

        if not action_match or not action_input_match:
            raise ValueError(f"Não foi possível analisar a saída do LLM: `{llm_output}`")

        action = action_match.group(1).strip()
        action_input = action_input_match.group(1).strip()

        # Tentar analisar a entrada como JSON
        try:
            # Verificar se a entrada já é um dicionário
            if isinstance(action_input, dict):
                return AgentAction(tool=action, tool_input=action_input, log=llm_output)

            # Tentar analisar a string como JSON
            parsed_input = json.loads(action_input)
            return AgentAction(tool=action, tool_input=parsed_input, log=llm_output)
        except json.JSONDecodeError:
            # Se não for um JSON válido, tentar extrair os valores diretamente usando regex
            try:
                a_match = re.search(r'"a":\s*(\d+(?:\.\d+)?)', action_input)
                b_match = re.search(r'"b":\s*(\d+(?:\.\d+)?)', action_input)

                if a_match and b_match:
                    a = float(a_match.group(1))
                    b = float(b_match.group(1))
                    return AgentAction(
                        tool=action, tool_input={"a": a, "b": b}, log=llm_output
                    )
            except Exception as e:
                print(e)

            # Se todas as tentativas falharem, retornar a entrada original
            return AgentAction(tool=action, tool_input=action_input, log=llm_output)


# Obter chaves de API das variáveis de ambiente
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
APITUBE_KEY = os.environ["APITUBE_KEY"]

# Definição das ferramentas personalizadas para o agente
@tool
def somar(a: float, b: float) -> float:
    """Soma dois números e retorna o resultado."""
    return a + b


@tool
def subtrair(a: float, b: float) -> float:
    """Subtrai o segundo número do primeiro e retorna o resultado."""
    return a - b


@tool
def noticias_cripto(word : str) -> str:
    """Retorna as últimas notícias sobre criptomoedas."""
    try:
        # API da apitube.io com limitações:
        # - Atraso de 12 horas nas notícias
        # - Máximo de 10 artigos por pedido
        # - Limite de 200 pedidos por dia
        response = requests.get(f"https://api.apitube.io/v1/news/everything?q=(title:{word}%20AND%20language.name:English)&is_duplicate=false&sort_by=publish_date&api_key={APITUBE_KEY}")
        if response.status_code == 200:
            data = response.json()
            news_results = data.get("results", [])
            # Extrair apenas os campos relevantes de cada notícia
            filtered_news = ""
            for item in news_results[:10]:  # Limitar a 10 notícias
                filtered_news += item.get('title', 'N/A') + "\n"
                filtered_news += item.get('published_at', 'N/A') + "\n"
                filtered_news += item.get('description', 'N/A') + "\n"
                filtered_news += item.get('body', 'N/A') + "\n\n"
            return filtered_news
        else:
            return "Erro ao buscar notícias: Status code " + str(response.status_code)
    except Exception as e:
        return f"Erro ao buscar notícias: {str(e)}"


# Configurar a ferramenta de pesquisa DuckDuckGo para obter informações da web
search_tool = DuckDuckGoSearchRun()

# Adicionar um nome e uma descrição mais detalhada para a ferramenta de pesquisa
search_tool.name = "search"
search_tool.description = """
Útil quando precisas de responder a perguntas sobre eventos atuais, informações gerais,
pessoas, lugares, factos, história, etc. Recebe uma consulta de pesquisa como entrada.
"""

# Lista de todas as ferramentas disponíveis para o agente
tools = [somar, subtrair, search_tool, noticias_cripto]

# Configurar o modelo de linguagem Groq (LLM)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Modelo Llama 3.3 de 70 biliões de parâmetros
    temperature=0.3,  # Temperatura baixa para respostas mais determinísticas
    groq_api_key=os.environ["GROQ_API_KEY"],  # Chave API da Groq
)

# Obter a data e hora atual formatada
current_datetime = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

# Definir o template do prompt para o agente
# Template com chaves escapadas corretamente para formatação
prompt = PromptTemplate.from_template("""
Você é um assistente útil que pode responder perguntas usando ferramentas quando necessário.
A data e hora atual é: {current_datetime}. Use esta informação quando relevante.

- Use a ferramenta 'somar' para operações matemáticas de adição.
- Use a ferramenta 'subtrair' para operações matemáticas de subtração.
- Use a ferramenta 'noticias_cripto' para obter notícias sobre criptos cript ativos, mundo das finanças.
- Use a ferramenta 'search' para buscar informações atuais, fatos, pessoas, eventos, etc.
- Para outras solicitações, responda diretamente sem usar ferramentas.

TOOLS:
{tools}

Para usar uma ferramenta matemática, utilize o seguinte formato:
Thought: Preciso usar uma ferramenta para resolver isso
Action: nome_da_ferramenta
Action Input: {{{{\"a\": número1, \"b\": número2}}}}

Para usar a ferramenta de noticias_cripto, utilize o seguinte formato:
Thought: Preciso buscar notícias sobre isso
Action: noticias_cripto
Action Input: {{{{word: palavra-chave}}}}

Para usar a ferramenta de pesquisa, utilize o seguinte formato:
Thought: Preciso buscar informações sobre isso
Action: search
Action Input: sua consulta de pesquisa aqui

IMPORTANTE: O Action Input para ferramentas matemáticas DEVE ser um objeto JSON válido. 
O Action Input para a ferramenta de pesquisa deve ser uma string de consulta.

O nome da ferramenta deve ser um destes: {tool_names}
Depois de usar uma ferramenta, você receberá um resultado no formato:
Observation: resultado da ferramenta

Use as informações da ferramenta para formular sua resposta. Se a informação não for suficiente, você pode usar outra ferramenta.

Quando tiver a resposta final, responda assim:
Thought: Agora sei a resposta
Final Answer: a resposta para a pergunta do usuário

USER INPUT: {input}

{agent_scratchpad}
""")

# Criar o agente ReAct com o modelo de linguagem, ferramentas e prompt definidos
agent = create_react_agent(llm, tools, prompt, output_parser=CustomOutputParser())

# Configurar o executor do agente
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,  # Não mostrar detalhes da execução
    handle_parsing_errors=True,  # Lidar com erros de análise
    max_iterations=5,  # Limitar número de tentativas para evitar loops infinitos
)

# Lista de exemplos de perguntas para testar o agente
exemplos = [
    "Quanto é 42 mais 28?",  # Teste da ferramenta de soma
    "Calcule 150 menos 75",  # Teste da ferramenta de subtração
    "Quem é o atual presidente do Portugal?",  # Teste da ferramenta de pesquisa
    "Quais são os principais pontos turísticos de Portugal em 2024?",  # Teste da ferramenta de pesquisa
    "Qual a diferença entre 2000 e 1985?",  # Teste da ferramenta de subtração
    "O que é machine learning?",  # Teste da ferramenta de pesquisa
    "Noticias da XRP e Ripple de 2025?",  # Teste da ferramenta de notícias
]

# Executar cada exemplo e apresentar os resultados formatados
for exemplo in exemplos:
    try:
        # Invocar o agente com a pergunta e a data/hora atual
        resposta = agent_executor.invoke(
            {"input": exemplo, "current_datetime": current_datetime}
        )
        # Criar um painel formatado com a resposta
        panel = Panel.fit(
            f"{resposta['output']}",
            title=f"{exemplo}",
            border_style="bold blue",
        )
        # Apresentar o painel no console
        console.print(panel)
        # Pausa de 2 segundos entre exemplos
        time.sleep(2)
    except Exception as e:
        print(f"ERRO: {str(e)}")
