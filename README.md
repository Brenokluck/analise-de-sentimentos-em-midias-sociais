# Analise de sentimentos em midias sociais

## Dataset

O data set contém 1,600,000 tweets extraido utilizando a api do twitter. Os tweets foram classificados como 0 (negativo) a 4 (positivo). O dataset contém 6 campos que são o alvo como integer, ids como integer, date como date, flag como string, user como string e text como string. Esses 6 campos são mostrados abaixo.

- alvo: A polaridade do tweet (0 - negativo, 2 - neutro, 4 - positivo)
- ids: O ID do tweet.
- data: A data do tweet.
- flag: A consulta. Se não houver consulta, este valor será NO_QUERY.
- usuário: O usuário que twittou.
- texto: O texto do tweet

Removemos os tweets com comprimento 0. Após esse processo, o conjunto de dados tem uma dimensão de 1592328×2.
As amostras positivas e negativas são iguais. A distribuição do conjunto de dados não apresenta assimetria, conforme mostrado abaixo.
![datadist](imgs/1.png)

# Pré-processamento

## Número de Letras

Apresentamos a frequência e a frequência relativa das letras em tweets completos. Por fim, aplicamos um teste qui-quadrado para verificar se a distribuição das letras nos tweets é semelhante à observada em textos em inglês.
![letterfreq](imgs/2.png)

Obtivemos o valor p (p) igual a 0, o que implica que a frequência das letras não segue a mesma distribuição que observamos nos testes em inglês, embora a correlação de Pearson seja muito alta (~96,7%).
| | Frequency | Expected |
|---------- |:-------------: |------: |
| frequency | 1.0 | 0.967421 |
| expected | 0.967421 | 1.0 |

Contamos o número de caracteres de cada tweet e analisamos o conjunto de dados considerando o número máximo de caracteres, o número mínimo de caracteres, a média da coluna de caracteres e seu desvio padrão. Nosso tweet mais longo tem 189 caracteres, o mais curto tem 1 caractere e a média do comprimento de todos os tweets é de 42,78 caracteres. O desvio padrão do comprimento de todos os tweets é de 24,16 caracteres.

## Número de Caracteres

Contamos o número de palavras em cada tweet e analisamos o conjunto de dados considerando o número máximo de palavras, o número mínimo de palavras, a média da coluna de número de palavras e seu desvio padrão. Nosso tweet mais longo tem 50 palavras, o mais curto tem 1 palavra e a média do comprimento de todos os tweets é 7,24. O desvio padrão do comprimento de todos os tweets é 4,03.

### Palavras mais comuns no conjunto de dados

![a](imgs/3.png)

## Tweets Positivos

### Palavras Mais Comuns em Tweets Positivos

![a](imgs/4.png)
![a](imgs/5.png)

## Tweets Negativos

### Palavras Mais Comuns em Tweets Negativos

![a](imgs/6.png)
![a](imgs/7.png)

## GloVe: Vetores Globais para Representação de Palavras

Podemos treinar o embedding nós mesmos. No entanto, essa abordagem pode levar muito tempo. Portanto, usamos a técnica de aprendizado por transferência e o GloVe: Vetores Globais para Representação de Palavras.
O algoritmo Vetores Globais para Representação de Palavras, ou GloVe, é uma extensão do método word2vec para aprendizado eficiente de vetores de palavras, desenvolvido por Pennington et al. em Stanford. É um algoritmo de aprendizado não supervisionado para obter representações vetoriais de palavras. O treinamento é realizado com base em estatísticas agregadas de coocorrência global de palavras de um corpus, e as representações resultantes exibem subestruturas lineares interessantes do espaço vetorial de palavras.
Baixamos o GloVe. Em seguida, inicializamos um índice de embedding com 400.000 vetores de palavras e uma matriz de embedding.

## Gráfico de Dispersão

Utilizamos métodos de extração de características, saco de palavras e incorporação de palavras.
O saco de palavras com TF-IDF é uma maneira comum e simples de extração de características
O saco de palavras é um modelo de representação de dados textuais e o TF-IDF é um método de cálculo
para pontuar a importância das palavras em um documento.
Após aplicar o saco de palavras com TF-IDF, criamos o gráfico de dispersão de acordo com
esses resultados.

### Gráfico de dispersão que mostra a correlação de palavras no corpus: vermelho indica palavras negativas, azul indica palavras positivas.

![a](imgs/8.png)

## Resultados do Pré-processamento

Exploramos nosso conjunto de dados aplicando algumas análises aos
atributos e criamos gráficos relacionados. Nosso conjunto de dados possui 2 atributos, incluindo o
atributo de rótulo. Aplicamos essas análises a eles.
Exploramos os tweets observando as letras e palavras neles contidas. Primeiramente,
contamos as letras de todos os tweets e calculamos suas frequências. Em seguida,
comparamos a frequência das letras em nossos dados com a frequência esperada das
letras do alfabeto inglês. Embora haja algumas exceções, para a maioria das letras, as
frequências de nossos dados são muito próximas das esperadas.
O número de caracteres e palavras também foi contado e analisado.
O número mínimo de caracteres em todos os tweets é 1, enquanto o número máximo é 189. Como a média é em torno de 42 e o desvio padrão é em torno de 24, pode-se dizer
que um pequeno número de tweets possui um grande número de caracteres. Resultado semelhante
pode ser observado na análise de palavras. Ao contar o número de palavras, observa-se que
o número máximo de palavras em tweets é 50, enquanto o número mínimo é 1.
A média é em torno de 7 e o desvio padrão é em torno de 4, o que resulta em um comportamento semelhante ao do
número de caracteres. Um número muito pequeno de tweets apresenta um alto número de palavras.
De acordo com esses resultados, pode-se interpretar que ambos os gráficos, o do número de caracteres
e o do número de palavras, são assimétricos.
Após a contagem do número de palavras usadas nos tweets, o uso das palavras é analisado. Como as stopwords (palavras irrelevantes) são geralmente as palavras mais usadas nos textos e podem nos impedir de obter os resultados corretos, elas são calculadas filtrando as stopwords. Além disso, as palavras mais comuns para rótulos positivos e negativos são separadas. Em seguida, um gráfico de dispersão é obtido usando alguns métodos de extração de características. O gráfico mostra a correlação entre as palavras.

# Análise Preditiva

Para experimentos de classificação/regressão, a porcentagem do conjunto de teste foi definida como
20%. Seis modelos diferentes foram aplicados: CNN Modelo-1, CNN Modelo-2, LSTM
Modelo-1, LSTM Modelo-2, Naive Bayes Modelo-1 e Naive Bayes Modelo-2. Abaixo,
são apresentados a precisão, a revocação, a pontuação F1 e a acurácia dos modelos.

## Classificação/Regressão

- Modelo CNN 1: Conv1D = 64, Dense = 512, Dense = 512, Tamanho do Lote 1024
- Modelo CNN 2: Conv1D = 64, Dense = 512, Dense = 512, Tamanho do Lote 512
- Modelo LSTM 1: Tamanho do Lote 1024
- Modelo LSTM 2: Tamanho do Lote 512
- Modelo Naive Bayes Multinomial 1: Vetorizador de Contagem
- Modelo Naive Bayes Multinomial 1: TF-IDF

![a](imgs/9.png)
![a](imgs/10.png)
![a](imgs/11.png)

## Curvas ROC

Após determinar as métricas de avaliação, as curvas ROC dos modelos são
construídas. Os valores de AUC também são calculados e exibidos na parte inferior de cada gráfico.
![a](imgs/12.png)
![a](imgs/13.png)

## Matriz de Confusão

As matrizes de confusão dos 6 modelos usados ​​para treinar os dados, incluindo o modelo de melhor
desempenho, LSTM-1, são as seguintes:

![a](imgs/14.png)
![a](imgs/15.png)
![a](imgs/16.png)

## Análise de Significância Estatística

De acordo com a acurácia, P, R, F1 e AUC, nosso modelo de melhor desempenho é o LSTM
modelo 1 com tamanho de lote de 1024 e acurácia de 0,789, e o concorrente mais próximo do
modelo LSTM 1 é o modelo CNN 1 com acurácia de 0,781. O Naive Bayes Multinomial com
tf-idf é o algoritmo de pior desempenho entre eles, com acurácia de 0,758.

![a](imgs/17.png)

## Resultados da Análise Preditiva

Nosso conjunto de dados bruto possui características desnecessárias para o nosso propósito. Seu primeiro valor de entropia
foi 41,08. Em seguida, removemos as colunas desnecessárias, excluímos as linhas vazias
e obtivemos um valor de entropia de 14,73. Após esse pré-processamento,
podemos facilmente observar uma mudança significativa nos valores de entropia.

Após todos os seis experimentos, podemos ver que diferentes LSTM e CNN nos fornecem
taxas de precisão muito próximas após o treinamento. Embora as diferenças sejam realmente pequenas,
o Modelo LSTM-1 obteve o melhor resultado e os modelos Naive Bayes tiveram um desempenho ligeiramente
pior.

Os modelos Naive Bayes têm as melhores durações de tempo de treinamento. Eles têm uma velocidade muito boa
em comparação com os modelos LSTM e CNN. Os modelos LSTM-1, LSTM-2 e
CNN-1 têm tempos de treinamento próximos, pois cada época leva de 10 a 13 minutos para
esses modelos. Embora a alteração do tamanho do lote no LSTM não tenha apresentado uma diferença significativa nos resultados, o modelo CNN-2 tem um tempo de treinamento melhor, de cerca de 7 a 8 minutos por época. Além disso, sua precisão é muito próxima à dos demais.

O modelo LSTM-1 tem uma taxa de precisão de 78,9% com um tamanho de lote de 1024 e o modelo LSTM-2 tem uma taxa de precisão de 78,6% com um tamanho de lote de 512. O modelo CNN-1 tem uma taxa de precisão de 78,2% com um tamanho de lote de 1024 e o modelo CNN-2 tem uma taxa de precisão de 77,2% com um tamanho de lote de 512. Ambos os algoritmos têm tempos de treinamento melhores com um tamanho de lote de 512, são melhores do que seus modelos com tamanho de lote de 1024 e suas taxas de precisão são muito próximas.

Como resultado, podemos dizer que os modelos LSTM e CNN com tamanho de lote de 1024 são melhores em termos de taxa de precisão. No entanto, os modelos com tamanho de lote de 512 têm taxas de precisão próximas com tempos de treinamento melhores.
Para as taxas de precisão dos modelos Naive Bayes, há uma pequena diferença de cerca de
1,5%. Como resultado, podemos dizer que o Naive Bayes com o método CountVectorizer
apresenta melhores resultados do que o Naive Bayes com o método TF-IDF.
