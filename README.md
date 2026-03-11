# Projeto Recomendador de Filmes

Aplicacao local de recomendacao de filmes com backend em FastAPI, frontend em JavaScript modular e busca vetorial local com FAISS.

## Visao geral

Este projeto junta tres ideias que apareceram ao longo do modulo 1 do curso:

1. pipeline explicito de dados, treino e inferencia
2. separacao clara de responsabilidades entre interface, servicos e regras de negocio
3. uso de embeddings para transformar conteudo em vetores comparaveis por similaridade

O resultado pratico e um recomendador por conteudo que:

- baixa a base TMDB do Kaggle
- preprocessa filmes, elenco, direcao e reviews
- gera vetores densos com TensorFlow
- indexa os vetores em FAISS
- recomenda filmes por perfil local e historico de feedback persistido

## Como a logica do projeto conversa com o modulo 1

### Relacao com o Bloco de Notas do Curso

As anotacoes do notebook `Bloco_de_Notas_Curso.ipynb` descrevem uma sequencia de raciocinio que aparece inteira aqui:

- obtencao e validacao do dataset: o projeto baixa a base do Kaggle e organiza os CSVs em `backend/data/raw`
- separacao entre aquisicao, treino, exportacao e inferencia: isso aparece no backend em servicos diferentes e artefatos persistidos
- consistencia entre treino e inferencia: os mesmos pesos, escalas e reducoes usados no treino sao salvos no bundle e reaproveitados na recomendacao
- regras de decisao na saida: o frontend nao mostra apenas scores crus; ele incorpora curtidos, favoritos, descartados e mensagens de uso
- avaliacao e otimizacao pratica: o relatorio de treino registra loss, curva de validacao, hiperparametros e metadados do indice vetorial

### Comparacao com o Exemplo 0

No `exemplo-00-z`, o fluxo basico e:

1. transformar dados em tensor
2. treinar uma rede
3. chamar `predict`
4. interpretar a saida

Aqui a mesma logica foi expandida para um caso real:

- em vez de poucos atributos manuais, usamos blocos textuais e numericos do TMDB
- em vez de classificacao softmax, usamos autoencoder para aprender uma representacao vetorial compacta
- em vez de uma unica categoria prevista, usamos similaridade vetorial para rankear filmes
- em vez de imprimir probabilidades no terminal, expomos tudo por API e interface web

Em resumo: o Exemplo 0 ensina a mecanica de treino e inferencia; este projeto aplica a mesma mecanica em um pipeline de recomendacao com persistencia e busca vetorial.

### Comparacao com o Exemplo 1 de ecommerce

O exemplo de ecommerce do modulo 1 trabalha com controladores, servicos, views e eventos para separar interface, dados e treino. Este projeto segue a mesma linha:

- `frontend/controllers`: coordenam busca, treino e recomendacoes
- `frontend/services`: isolam acesso a API e persistencia local do perfil
- `frontend/views`: renderizam cards, detalhes, status e relatorios
- `backend/app/services`: concentram preprocessamento, treino, indexacao vetorial e recomendacao

Assim como no ecommerce existe um fluxo de treino desacoplado da interface, aqui o treino roda em background, atualiza status e publica relatorio consumido pelo frontend.

### Comparacao com os exemplos de embeddings

Nos exemplos de embeddings, a ideia central e converter texto ou objetos em vetores e depois comparar esses vetores. E exatamente isso que ja fazemos aqui:

- o encoder do autoencoder gera embeddings densos de 64 dimensoes
- os embeddings sao normalizados
- o FAISS salva um indice vetorial local em `backend/artifacts/movie_vectors.faiss`
- a recomendacao usa busca por similaridade no indice para recuperar candidatos proximos

Ou seja: a parte vetorial nao e um passo futuro. Ela ja faz parte da arquitetura atual.

## Arquitetura do projeto

```text
ProjetoRecomendadordeFilmes/
  backend/
    app/
      main.py
      routes/
      services/
      models/
    data/
      raw/
      processed/
    artifacts/
    scripts/
  frontend/
    controllers/
    services/
    views/
  notebooks/
```

## Logica completa do backend

### 1. Download e organizacao da base

O `DatasetService` baixa a base TMDB via Kaggle e copia os CSVs relevantes para `backend/data/raw`.

Arquivos principais usados no pipeline:

- `movies.csv`
- `cast.csv`
- `crew.csv`
- `reviews.csv`

### 2. Preprocessamento

O `PreprocessingService` monta um dataframe consolidado por filme.

O que ele faz:

- padroniza `movie_id`
- escolhe colunas equivalentes mesmo quando o CSV muda de nome
- agrega ate 6 nomes de elenco por filme
- agrega ate 3 diretores
- agrega reviews de forma resumida e menos ruidosa
- limpa HTML, links, caracteres estranhos e excesso de espacos
- reduz o tamanho de overview e review para evitar ruido exagerado

Saida principal:

- `backend/data/processed/movies_processed.csv`

### 3. Montagem das features

O `ModelTrainingService` monta features em blocos:

- texto: `title`, `overview`, `genres_text`, `cast_text`, `director_text`, `review_text`
- numericos: `vote_average`, `popularity`, `runtime`, `release_year`, `review_score`, `review_count`

Cada bloco textual passa por TF-IDF com peso proprio. Depois:

1. os blocos sao concatenados
2. os numericos sao escalados
3. tudo passa por `TruncatedSVD`
4. o vetor reduzido vai para o autoencoder

Importante: nesta revisao o projeto passou a usar embedding de 64 dimensoes e reviews mais curtas e leves para reduzir ruido.

### 4. Treino do modelo

O modelo e um autoencoder denso em TensorFlow/Keras.

Objetivo do treino:

- receber o vetor reduzido do filme
- comprimir esse vetor em um embedding menor
- reconstruir a entrada original

O gargalo da rede e justamente o embedding que depois sera usado para recomendacao.

### 5. Indexacao vetorial

Depois do treino:

- o encoder gera embeddings para todos os filmes
- os vetores sao normalizados
- o FAISS cria um indice `IndexFlatIP`
- o indice e salvo localmente

Artefatos principais:

- `backend/artifacts/movie_recommender.joblib`
- `backend/artifacts/movie_embeddings.npy`
- `backend/artifacts/movie_vectors.faiss`
- `backend/artifacts/training_report.json`

### 6. Recomendacao

O `RecommendationService` suporta dois fluxos:

- recomendacao por filme
- recomendacao por perfil

No perfil, a logica atual:

1. junta likes e favorites como sinais positivos
2. usa dislikes como penalizacao
3. mescla o perfil atual do frontend com o feedback salvo no backend para o usuario
4. aplica pesos diferentes por acao e um pequeno ganho por recencia
5. monta um vetor medio do perfil
6. consulta o indice vetorial
7. remove itens ja vistos
8. reordena resultados com pesos extras de intersecao de elenco

Isso produz uma recomendacao por conteudo enriquecida por heuristicas simples de perfil.

Na pratica, isso significa que o sistema nao depende apenas da sessao atual do navegador. Se o usuario ja curtiu, favoritou ou descartou filmes antes, esse historico salvo passa a influenciar a recomendacao seguinte.

## Logica completa do frontend

O frontend foi organizado no mesmo espirito dos exemplos do curso.

### Controllers

- `SearchController`: cuida da busca textual
- `RecommendationController`: administra perfil local, feedback e recomendacoes
- `TrainingController`: chama download, treino e atualizacao de status
- `AppController`: conecta tudo

### Services

- `ApiService`: centraliza chamadas HTTP
- `SessionService`: persiste likes, dislikes e favoritos no `localStorage`

### Views

- `MovieListView`: cards da busca e recomendacoes
- `MovieDetailsView`: detalhe do filme e botoes principais
- `TrainingView`: status, graficos e resumo do treino

## Fluxo ponta a ponta

1. O usuario baixa a base.
2. O backend preprocessa os CSVs.
3. O TensorFlow treina o autoencoder.
4. O encoder gera embeddings.
5. O FAISS cria o indice vetorial.
6. O frontend consulta busca, detalhes e recomendacoes.
7. Likes, favoritos e descartes refinam o perfil local.
8. O historico de feedback persistido e reutilizado para reforcar sinais de preferencia entre sessoes.

## Sobre loss, qualidade e leitura das metricas

O projeto usa `mse` como loss de reconstrucao, nao acuracia de classificacao.

Isso significa que:

- loss mais baixo e melhor
- mas o numero nao deve ser lido como porcentagem de erro
- ele mede o quanto o autoencoder conseguiu reconstruir a representacao do filme

Tambem mostramos uma `precisao estimada`, derivada da validacao do autoencoder. Ela ajuda a acompanhar estabilidade de treino, mas nao substitui metricas reais de recomendacao.

Nesta etapa, o projeto tambem passou a calcular metricas offline de ranking com base no historico de feedback persistido:

- Precision@5
- Recall@5
- NDCG@5
- Precision@10
- Recall@10
- NDCG@10
- MRR

A avaliacao segue um esquema leave-one-out simples:

1. para cada usuario com feedback suficiente, separamos o ultimo item positivo como alvo oculto
2. usamos o restante do historico para montar o perfil
3. geramos o ranking no espaco vetorial
4. verificamos em que posicao o item oculto reaparece

Isso e muito mais informativo do que olhar apenas a loss do autoencoder, porque aproxima a avaliacao do comportamento real do recomendador.

### Como interpretar cada metrica

- `Precision@5`: entre os 5 primeiros itens recomendados, qual fracao realmente continha o item relevante esperado. Quanto maior, melhor a qualidade do topo da lista.
- `Recall@5`: verifica se o item relevante apareceu entre os 5 primeiros. No nosso cenario leave-one-out com um item alvo, ele funciona como taxa de acerto no top 5.
- `NDCG@5`: mede nao apenas se acertou, mas quao cedo o item relevante apareceu. Acertar em primeiro vale mais do que acertar em quinto.
- `Precision@10`: mesmo raciocinio da Precision@5, mas olhando uma janela maior. Ajuda a ver se o modelo continua bom quando a lista cresce.
- `Recall@10`: mostra a chance de o item oculto aparecer entre os 10 primeiros.
- `NDCG@10`: mede a qualidade da ordem dos 10 primeiros resultados.
- `MRR`: media do inverso da posicao do primeiro item relevante. Se o alvo aparece sempre entre as primeiras posicoes, o MRR sobe rapidamente.

Leitura pratica:

- loss melhorando com ranking pior significa que a rede reconstruiu melhor os vetores, mas isso nao se traduziu em recomendacao mais util
- ranking melhorando mesmo com pequena piora de loss pode ser aceitavel, porque o objetivo final do sistema e recomendar melhor
- `NDCG` e `MRR` costumam ser as metricas mais uteis para avaliar a qualidade do topo da lista

## Como rodar

### Backend

```bash
cd backend
pip install -r requirements.txt
python scripts/download_dataset.py
python scripts/train_model.py
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm start
```

## Melhorias futuras mais importantes

- enriquecer a avaliacao offline com mais usuarios e janelas temporais melhores
- testar diferentes tamanhos de embedding e pesos por bloco
- introduzir feedback implicito e historico real de usuarios
- considerar um banco vetorial dedicado se o projeto precisar de filtros e escala maiores