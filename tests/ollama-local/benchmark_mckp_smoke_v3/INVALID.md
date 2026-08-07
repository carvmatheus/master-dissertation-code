# Invalid artifact layout

Os casos executados nesta rodada eram oficiais, mas um caminho relativo foi
resolvido a partir de diretórios diferentes pelo orquestrador e pelo runner.
O manifesto e o log ficaram nesta raiz, enquanto os resultados foram gravados
fora do repositório. A rodada posterior usa um caminho absoluto e deve ser
adotada para inspeção e reprodução.
