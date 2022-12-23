# Scripts e Modelos da Dissertação de Mestrado

## Introdução

Nesse repositório estão presentes todos os scripts, módulos e modelos utilizados na dissertação de mestrado de título **Análise do desempenho termodinâmico de um sistema de trigeração para produção de potência, refrigeração e água dessalinizada**, elaborada por Arthur Pontes de Miranda Ramos Soares e apresentada à Faculdade de Engenharia Química da Universidade Estadual de Campinas (UNICAMP).

O objetivo é deixar públicos todos os modelos elaborados utilizando o _software_ Engineering Equation Solver (EES) e os scripts escritos em Python destinados à análise paramétrica, otimização e geração de gráficos.

## Utilização

Para modificar ou utilizar o código, basta baixar o repositório no formado .zip ou clonar o repositório. Além disso, é necessário possuir o EES instalado para execução dos programas referentes à análise paramétrica e otimização (requer licença comercial).

**IMPORTANTE**: É necessário especificar o `path` para o executável do EES (EES.exe) em `scripts\config.py`:

```Python
## scripts/config.py
EES_exe = r"C:\sua\pasta\aqui\EES.exe"
```

**OBS:** Em `scripts\config.py` também é possível especificar uma pasta de destino para resultados de alguns scripts. Todavia, para a otimização e para a análise paramétrica, os resultados são destinados à pasta `results`, criada na mesma pasta em que os modelos do EES estão.

**IMPORTANTE**: Ao extrair ou clonar o repositório, execute os scripts da pasta mãe. Caso contrário, as dependências presentes na pasta `ees` não serão encontradas.

Especificação de versões:

- Windows 10,
- EES: Profissional V10.561,
- Python: 3.7.11

Dependências externas:

- Numpy: 1.20.3,
- matplotlib: 3.5.1,
- Pandas: 1.3.2,
- DEAP: 1.3.1,
- Fonte _computer modern_ (Copyright (C) 2003-2009, Andrey V. Panov (panov@canopus.iacp.dvo.ru), with Reserved Font Family Name "Computer Modern Unicode fonts").

## Documentação

O presente repositório está dividido em três partes principais:

- `ees`: Módulo que contém as definições das classes e funções destinadas a efetuar a avaliação, análise paramétrica e otimização dos modelos do EES, utilizando o Python.
- `models`: Pasta que contém os modelos do EES referentes à dissertação;
- `scripts`: Pasta que contém os scripts utilizados para obtenção dos resultados. Esses scripts dependem do código definido em `ees`.

Os quatro principais objetivos do código são:

- Avaliar os modelos do EES a partir do Python,
- Realizar a análise paramétrica dos principais parâmetros operacionais, de forma automatizada, utilizando o Python,
- Realizar a otimização dos modelos utilizando o Algoritmo Genérico, implementado a partir da biblioteca [DEAP](https://deap.readthedocs.io/en/master/),
- Gerar gráficos.

### Avaliar os modelos

Nesse caso, isso significa fazer com que o Python abra uma instância do EES, forneça valores para as variáveis de entrada do modelo, envie um comando para que o EES execute a resolução do modelo, receba os resultados e processe-os, salvando-os na forma de arquivos.

Para isso, foi utilizada a funcionalidade de Macro do EES, em que um arquivo macro (.emf) pode ser construído e passado para o EES via linha de comando, sendo esse arquivo macro, composto por comandos que o EES deve executar em sequência.

Um exemplo da avaliação dos modelos está presente no arquivo `scripts/others/solve_model.py`.

Os resultados são obtidos a utilizando a classe `SolveModel` que é importada de `ees.solvemodel`.

`SolveModel` requer:

- O path para o executável do EES,
- O path para o modelo do EES,
- Um dicionário de inputs, em que as chaves são os nomes das variáveis como presentes no modelo e os valores são os valores desejados das variáveis,
- Uma lista de outputs, ou seja, variáveis de saída que serão retornadas pelo EES, também como escritas nos modelos,
- E um identificador de execução (runID), que é uma forma de diferenciar diferentes rodadas de execução (evita sobscrever os resultados anteriores).

```Python
from scripts.config import EES_exe
from ees.solvemodel import SolveModel

EES_model = "path/to/model.EES"

# É necessário que essas variáveis estejam comentadas (removidas da execução) em model.EES.
inputs = {
    "P": 101.325,
    "T_1": 298.15,
    "T_2": 300.15,
    "m": 0.05,
}

outputs = ["h", "s", "Ex", "V"]
SolveModel(EES_exe, EES_model, inputs, outputs, runID=runID)
```

### Análise paramétrica

Nesse caso, foi utilizada a mesma lógica da avaliação dos modelos, entretanto a classe responsável, `ParametricStudies` (importada de ees.parametric), constrói um arquivo macro que engloba a avaliação dos modelos $n \times m$ vezes, em que $n$ é o número de parâmetros a serem avaliados e $m$ o número de valores avaliados para determinado parâmetro.

`ParametricStudies` requer:

- O path para o executável do EES,
- O path para o modelo do EES,
- Um dicionário de inputs,
- Um dicinário das faixas a serem avaliadas, em que as chaves são as variáveis, e os valores são lista de valores a serem avaliados para determinada variável,
- Uma lista de outputs,
- E um identificador de execução (runID).

```Python
from ees.parametric import ParametricStudies
from scripts.config import EES_exe

EES_model = "path/to/model.EES"

# É necessário que essas variáveis estejam comentadas (removidas da execução) em model.EES.
inputs = {
    "P": 101.325,
    "T_1": 298.15,
    "T_2": 300.15,
    "m": 0.05,
}

# T_2 e P serão avaliados
parametric_inputs = {
    "T_2": [280, 290, 300, 310, 320, 330],
    "P": [100, 200, 300, 400, 500],
}

outputs = ["h", "s", "Ex", "V"]
ParametricStudies(EES_exe, EES_model, inputs, parametric_inputs, outputs, run_id="param_analysis_v1")
```

Exemplos de scripts para análise paramétrica estão presentes em `scripts/parametric/parametric_study.py`

### Otimização

Quanto à otimização utilizando Algoritmo Genético, não foi possível utilizar o método dos arquivos macro. Isso porque o Algoritmo precisa saber a aptidão de cada possível resposta durante a execução para determinar qual a melhor. Portanto, foi necessário utilizar uma conexão DDE.

DDE, ou _Dynamic Data Exchange_, é um protocolo de transferência de dados entre aplicações no Windows. Com isso, é possível que um o Python envie comandos para serem executados no EES, como: "Abrir arquivo modelo.EES", "importar variavel_1, variabel_2 do clipboard", "avaliar modelo", "exportar variavel_3, variavel_4 para clipboard".

Utilizando as funcionalidades de importar e exportar para o clipboard, é possível fornecer e receber dados provenientes de cada execução do EES de forma dinâmica.

Para isso é utilizada a classe `GAOptimizationStudy`, importada de ees.optimization. Essa classe requer:

- O path para o executável do EES,
- O path para o modelo do EES,
- Um dicionário de inputs,
- Uma lista de outputs,
- Um identificador de execução (runID),
- Um dicionário das faixas de operação das variáveis de decisão, em que as chaves são os nomes das variáveis e os valores são tuplas, cujo primeior valor é o limite inferior e o segundo é o limite superior,
- Um dicionário da variável alvo, especificando a variável alvo, sua _string_ de display e o tipo de otimização (max ou min),
- E um dicionário das configurações do GA (ver exemplo).

```Python
from scripts.config import EES_exe
from ees.optimization import GAOptimizationStudy

EES_model = "path/to/model.EES"

# É necessário que essas variáveis estejam comentadas (removidas da execução) em model.EES.
inputs = {
    "P": 101.325,
    "T_1": 298.15,
    "T_2": 300.15,
    "m": 0.05,
}

outputs = ["h", "s", "Ex", "V", "efficiency"]

decision_variables = {
    "T_1": (275.15, 350),
    "m": (0.01, 0.10)
}

target_variable = {
    "target_variable": "efficiency", 
    "target_variable_display": r"$ \eta_{sys} $", 
    "problem": "max"
}

ga_config = {
    'seed': 5,
    'population': 10,
    'crossover': {'rate': 0.4, 'method': 'cxBlend', 'params': {'alpha': 0.4}},
    'mutation': {'rate': 0.15, 'method': 'mutPolynomialBounded',
                    'params': {'indpb': 0.15, 'low': low, 'up': up, 'eta': 3}},
    'selection': {'method': 'selTournament', 'params': {'tournsize': 7}},
    'max_generation': 5,
    'cvrg_tolerance': 1e-05,
    'verbose': True
}

eesopt = GAOptimizationStudy(EES_exe, EES_model, inputs, outputs, runID=runID)
eesopt.set_decision_variables(decision_variables)
eesopt.set_target_variable(**target_variable)
eesopt.execute(ga_config)
```

Exemplos completos de scripts para otimização estão presentes em `scripts/optimization/optimization_cases.py`
