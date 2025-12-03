# Analise de sentimentos em midias sociais

## Dataset

Foi usado um data set público contém 1,600,000 tweets extraido utilizando a api do twitter. Os tweets foram classificados como 0 (negativo) a 4 (positivo). O dataset contém 6 campos que são o alvo como integer, ids como integer, date como date, flag como string, user como string e text como string. Esses 6 campos são mostrados abaixo.

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

## Quantidade de Letras

Apresentamos a frequência e a frequência relativa das letras em tweets completos. Por fim, aplicamos um teste qui-quadrado para verificar se a distribuição das letras nos tweets é semelhante à observada em textos em inglês.
![letterfreq](imgs/2.png)

Obtivemos o valor p (p) igual a 0, o que implica que a frequência das letras não segue a mesma distribuição que observamos nos testes em inglês.
| | Frequency | Expected |
|---------- |:-------------: |------: |
| frequency | 1.0 | 0.967421 |
| expected | 0.967421 | 1.0 |

Contamos o número de caracteres de cada tweet e analisamos o conjunto de dados considerando o número máximo de caracteres, o número mínimo de caracteres, a média da coluna de caracteres e seu desvio padrão. Nosso tweet mais longo tem 189 caracteres, o mais curto tem 1 caractere e a média do comprimento de todos os tweets é de 42,78 caracteres. O desvio padrão do comprimento de todos os tweets é de 24,16 caracteres.

## Quantidade de Caracteres

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

## Matriz de Confusão

As matrizes de confusão dos 6 modelos usados ​​para treinar os dados, incluindo o modelo de melhor
desempenho, LSTM-1, são as seguintes:

![a](imgs/16.png)

## Análise de Significância Estatística

O modelo com melhor desempenho é o LSTM 1, que obteve acurácia de 0,789 usando tamanho de lote 1024. O segundo melhor é o CNN 1, com acurácia de 0,781. Já o pior resultado foi do Naive Bayes Multinomial com tf-idf, que alcançou acurácia de 0,758.

![a](imgs/17.png)

## Resultados da Análise Preditiva

Primeiro, nosso conjunto de dados tinha muitas informações que não eram úteis. A incerteza inicial era 41,08. Depois de remover colunas desnecessárias e excluir linhas vazias, a incerteza caiu para 14,73, mostrando uma grande melhoria após o pré-processamento.

Após os seis experimentos, vimos que os modelos LSTM e CNN tiveram acurácias muito parecidas. Mesmo assim, o LSTM-1 foi o melhor, enquanto os modelos Naive Bayes tiveram um desempenho um pouco pior.

Em compensação, os Naive Bayes são os mais rápidos para treinar, bem mais velozes do que LSTM e CNN. Os modelos LSTM-1, LSTM-2 e CNN-1 levam de 10 a 13 minutos por época, enquanto o CNN-2 é mais rápido, demorando 7 a 8 minutos, com acurácia parecida.

As acurácias ficaram assim:

LSTM-1: 78,9% (batch 1024)

LSTM-2: 78,6% (batch 512)

CNN-1: 78,2% (batch 1024)

CNN-2: 77,2% (batch 512)

Modelos com batch 512 são mais rápidos, enquanto os de 1024 têm acurácia um pouco maior.

No caso do Naive Bayes, a diferença entre os modelos foi pequena (cerca de 1,5%). Mesmo assim, o Naive Bayes com CountVectorizer teve desempenho melhor do que o Naive Bayes com TF-IDF.
