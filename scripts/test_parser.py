from src import query_parser

tests = [
    "quero um RPG de ação cooperativo até 50 conto pra jogar com 2 amigos no Windows",
    "jogo levinho de tiro pra jogar com a rapaziada sem gastar nada",
    "jogo relaxante de fazenda com música bonita",
    "souls-like desafiador no linux",
]

for t in tests:
    print("=" * 60)
    print(t)
    try:
        r = query_parser.parse_query_dict(t)
        print(r)
    except Exception as e:
        print("ERR", e)
    print()
