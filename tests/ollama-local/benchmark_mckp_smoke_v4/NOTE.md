# Partial smoke only

Esta rodada valida Raw e o MCKP sobre uma amostra oficial do LongBench, mas o
controle uniforme ainda usava uma opção por partição e tornou-se inviável sob o
orçamento, escolhendo omissão total. A implementação posterior aplica o
compressor uniforme ao contexto completo como uma única unidade. Portanto, o
resultado do controle em `smoke_v4` não deve ser usado como evidência.
