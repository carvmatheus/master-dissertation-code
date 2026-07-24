# Ollama benchmark runs

Artefatos dos benchmarks locais executados sequencialmente pelo Ollama.

- `models/`: metadados de cada modelo, incluindo a janela máxima declarada.
- `runs/<modelo>/ctx-<tokens>/`: CSV e JSON gerados por cada execução.
- `ollama-server.log`: log do servidor, quando o orquestrador precisar iniciá-lo.

Os pesos continuam em `~/.ollama`, sob controle do Ollama. Esta pasta guarda
somente metadados e resultados, evitando duplicar dezenas de gigabytes dentro
do repositório e do iCloud.

