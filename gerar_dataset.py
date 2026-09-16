"""
Gera manifestacoes.json (40 manifestações) e duplicatas_gabarito.json.

O enunciado diz que o arquivo seria fornecido, mas ele não veio junto com o
desafio. Este script recria um dataset que respeita todas as restrições:
  - ids M001..M040, data AAAA-MM-DD, 5 categorias oficiais
  - textos entre 50 e 800 caracteres
  - ~15% de duplicatas semânticas (6 pares = 12 manifestações envolvidas)
  - 5 manifestações longas (> 500 caracteres) para o chunking
  - M003 x M017 (buraco na Av. Brasil), M008 x M022 (posto sem médico / PSF)
    e M031 (lâmpada queimada na praça), exatamente como o enunciado cita

Uso: python gerar_dataset.py
"""
import json
from datetime import date, timedelta

I, S, G, E, A = "infraestrutura", "saúde", "segurança", "educação", "meio ambiente"

MANIFESTACOES = [
    # id, categoria, texto
    ("M001", I, "A rua Coronel Amaral está sem asfalto há meses. Quando chove vira um lamaçal e os carros atolam. Pedimos providência da prefeitura urgentemente."),
    ("M002", S, "Fui ao hospital municipal com meu filho com febre alta e esperamos mais de seis horas para ser atendidos. Não havia pediatra de plantão."),
    ("M003", I, "Tem um buraco enorme na Av. Brasil, na altura do número 1200, perto do supermercado. Vários carros já furaram o pneu e uma moto caiu semana passada."),
    ("M004", A, "O córrego que passa atrás do conjunto Jardim das Flores está cheio de esgoto e o mau cheiro é insuportável, principalmente à noite."),
    ("M005", G, "A rua onde moro, no bairro São José, está completamente escura porque os postes não acendem. Já tivemos dois assaltos no mês passado por causa disso."),
    ("M006", S, "Escrevo para relatar uma situação que se repete no posto de saúde do bairro Cruz das Armas. Minha mãe, de 72 anos, é hipertensa e diabética e depende dos remédios de uso contínuo da farmácia básica. Nos últimos três meses, todas as vezes que fomos buscar a medicação faltava pelo menos um item: ora a losartana, ora a metformina, ora a insulina. Os atendentes dizem que o estoque não chega e que devemos voltar em uma semana, mas a situação é sempre a mesma. Ela não tem condições de comprar tudo na farmácia particular e já teve dois episódios de pressão alta por interromper o tratamento. Além disso, a agenda do clínico geral só abre a cada dois meses e as vagas acabam em minutos. Peço que a Secretaria de Saúde verifique o abastecimento da farmácia e a oferta de consultas."),
    ("M007", E, "A creche municipal do bairro Mandacaru está sem merenda desde o início do mês. As crianças ficam o dia todo e só recebem biscoito."),
    ("M008", S, "O posto de saúde do bairro Geisel está sem médico há duas semanas. Chegamos às 5 da manhã, pegamos ficha e depois avisam que não vai ter atendimento."),
    ("M009", A, "Estão despejando entulho e lixo de construção no terreno baldio da rua das Acácias. Já virou ponto de lixo e tem rato e escorpião."),
    ("M010", G, "Os semáforos do cruzamento da Epitácio com a Ruy Carneiro ficam piscando em amarelo depois das 22h e os motoristas não respeitam. Já houve batidas."),
    ("M011", A, "O caminhão do lixo não passa na rua Padre Meira há dez dias. Os sacos estão acumulados na calçada, com mau cheiro e atraindo cachorros e urubus."),
    ("M012", I, "A calçada da rua Bancário Sérgio Guerra está toda quebrada. Minha avó usa cadeira de rodas e não consegue passar, precisa ir pela pista."),
    ("M013", E, "A escola municipal João Pessoa está com o teto da quadra caindo. As aulas de educação física foram suspensas e ninguém dá previsão de reforma."),
    ("M014", E, "A turma do 5º ano da escola municipal do bairro Valentina está sem professora de português desde março. Os alunos ficam na biblioteca sem aula."),
    ("M015", G, "Venho relatar o que acontece diariamente na praça central do bairro Bessa, entre a rua dos Coqueiros e a avenida da praia. Desde o começo do ano um grupo se instalou nos bancos da praça e passa o dia consumindo bebida alcoólica, abordando quem passa para pedir dinheiro e, em alguns casos, ameaçando moradores. As mães que levavam as crianças ao parquinho deixaram de frequentar o local. À noite piora porque metade dos refletores está queimada e a praça fica escura. Já registramos boletim de ocorrência duas vezes e ligamos para a Guarda Municipal, mas a viatura passa, o grupo se dispersa e volta quinze minutos depois. Os comerciantes já fecham mais cedo. Solicitamos ronda fixa da Guarda à noite, conserto da iluminação e instalação de câmeras de monitoramento na praça."),
    ("M016", A, "Uma árvore de grande porte na rua Silvino Chaves está com o tronco rachado e ameaça cair sobre a rede elétrica e as casas. Já solicitamos poda há dois meses."),
    ("M017", I, "O asfalto da avenida principal está todo esburacado, principalmente na Av. Brasil perto do mercado. Os buracos estão danificando os carros e causando acidentes de moto."),
    ("M018", S, "Liguei para o SAMU quando meu pai passou mal e a ambulância demorou mais de uma hora para chegar. Ele teve um AVC e a demora pode ter piorado o quadro."),
    ("M019", G, "Fui assaltada no ponto de ônibus da avenida Pedro II, em frente à farmácia, por volta das 19h. Dois homens em uma moto levaram meu celular. É o terceiro caso ali este mês."),
    ("M020", E, "O ônibus escolar que leva as crianças da zona rural para a escola do distrito está quebrado há três semanas e ninguém providenciou substituto."),
    ("M021", I, "A ponte de madeira sobre o riacho na comunidade do Roger está com tábuas soltas e podres. As crianças passam por ali todo dia para ir à escola."),
    ("M022", S, "Falta atendimento no PSF do Geisel. Faz quinze dias que nenhum profissional aparece na unidade; a gente madruga na fila, pega senha e volta pra casa sem consulta."),
    ("M023", A, "A queima de lixo no quintal de uma casa na rua Josefa Taveira acontece toda tarde. A fumaça entra nas casas vizinhas e minha filha tem asma."),
    ("M024", I, "Registro uma reclamação sobre a drenagem do bairro Bancários, nas ruas em torno da praça da Paz. Toda vez que chove mais forte as galerias pluviais não dão conta e a água sobe rápido, invadindo garagens e casas térreas. Na última chuva, em agosto, a água chegou a quase meio metro dentro da minha casa e perdi geladeira, sofá e colchões. Os bueiros estão entupidos com lixo e areia e não vejo limpeza há mais de um ano. A obra de ampliação da galeria iniciada no ano passado foi abandonada e deixou valas abertas na rua, que acumulam água e viram criadouro de mosquito. Peço a limpeza urgente dos bueiros e bocas de lobo, a retomada da obra e um plano de contingência para o período chuvoso, porque o fim do ano se aproxima e o problema só piora."),
    ("M025", S, "A UPA do Oitizeiro está sem aparelho de raio-x funcionando. Fui encaminhado para outra unidade do outro lado da cidade com o braço quebrado."),
    ("M026", E, "Os alunos do quinto ano da escola do Valentina estão há meses sem aula de português porque não tem professor. Os filhos da gente ficam parados na biblioteca."),
    ("M027", I, "A rotatória da entrada do bairro Altiplano está sem sinalização e sem faixa de pedestre. Atravessar ali é muito perigoso, ainda mais para idosos."),
    ("M028", G, "Iluminação pública sem funcionar em toda a rua Manoel Deodato, no São José. Está escuro demais à noite e os moradores têm medo de sair, já ocorreram assaltos."),
    ("M029", A, "O parque Sólon de Lucena está com os lagos sujos, cheios de lixo e algas. Os peixes estão morrendo e o cheiro está forte."),
    ("M030", E, "A biblioteca da escola municipal do bairro Cristo Redentor está fechada há um semestre porque não há funcionário para atender. Os livros estão mofando."),
    ("M031", I, "A lâmpada do poste da praça do bairro Torre está queimada há mais de um mês. A praça fica toda escura e as pessoas pararam de frequentar à noite."),
    ("M032", S, "Marquei consulta com cardiologista pelo SUS em fevereiro e só consegui vaga para novembro. Nove meses de espera para uma consulta é um absurdo."),
    ("M033", E, "Sou mãe de dois alunos da Escola Municipal Frei Damião, no bairro Funcionários, e quero registrar a situação precária da escola. Os banheiros dos alunos estão interditados desde o meio do ano porque o encanamento estourou, e as crianças usam o banheiro dos professores em revezamento, gerando filas enormes; muitas evitam beber água para não precisar ir. Os ventiladores não funcionam e, com calor de quase 35 graus, os alunos ficam sonolentos e passam mal. Faltam carteiras, então alguns assistem aula em pé ou dividindo cadeira. A quadra não tem cobertura e a educação física acontece sob sol forte ao meio-dia. A direção sempre diz que o pedido foi encaminhado à Secretaria de Educação, mas nada muda. Pedimos vistoria e um cronograma real de reparos, começando pelos banheiros e pela ventilação."),
    ("M034", G, "Motos fazem racha na avenida Beira Rio todas as sextas-feiras à noite. O barulho é ensurdecedor e já teve acidente com pedestre."),
    ("M035", A, "Coleta de lixo não está acontecendo na rua Padre Meira. Faz quase duas semanas que o caminhão não aparece e o lixo está acumulado nas calçadas, com cheiro horrível e animais mexendo."),
    ("M036", I, "O abastecimento de água no bairro Colinas do Sul falha toda semana. Ficamos de quinta a domingo sem água na torneira e a caixa não dá conta."),
    ("M037", S, "O agente de saúde não visita nossa rua no bairro Alto do Mateus há meses. Tem caso de dengue na vizinhança e ninguém apareceu para orientar."),
    ("M038", G, "Roubos frequentes na parada de ônibus da Pedro II perto da farmácia, sempre no fim da tarde. Bandidos de moto levam celular das pessoas que esperam o ônibus. Precisamos de policiamento."),
    ("M039", E, "Faltam vagas em creche no bairro Grotão. Estou na fila de espera há um ano e meio e preciso trabalhar, não tenho com quem deixar meu filho."),
    ("M040", A, "Quero denunciar a poluição sonora e ambiental de uma fábrica de blocos de concreto instalada em área residencial no bairro Mangabeira, na rua Doutor José Alves. A fábrica funciona de segunda a sábado desde as seis da manhã, com máquinas que produzem um barulho contínuo insuportável dentro de casa, além do pó de cimento que cobre calçadas e carros e entra pelas janelas. Vários moradores desenvolveram problemas respiratórios e uma vizinha idosa foi internada com crise de bronquite. Os caminhões de blocos estacionam na calçada e já quebraram a tubulação de água da rua duas vezes. A Secretaria de Meio Ambiente disse que a empresa tem licença, mas não acreditamos que ela permita esse nível de barulho e poeira em área residencial. Pedimos fiscalização e medição de ruído."),
]

# pares de duplicatas semânticas (gabarito para a Entrega 2)
DUPLICATAS = [
    ["M003", "M017"],  # buraco na Av. Brasil
    ["M008", "M022"],  # posto do Geisel sem médico
    ["M005", "M028"],  # rua escura no São José
    ["M011", "M035"],  # lixo acumulado na rua Padre Meira
    ["M014", "M026"],  # sem professor de português no Valentina
    ["M019", "M038"],  # assaltos na parada da Pedro II
]

if __name__ == "__main__":
    inicio = date(2026, 6, 1)
    registros = []
    for k, (mid, cat, txt) in enumerate(MANIFESTACOES):
        assert 50 <= len(txt) <= 800, (mid, len(txt))
        registros.append({
            "id": mid,
            "data": (inicio + timedelta(days=(k * 3) % 92)).isoformat(),
            "categoria_oficial": cat,
            "texto": txt,
        })

    assert [r["id"] for r in registros] == [f"M{i:03d}" for i in range(1, 41)]
    longas = [r["id"] for r in registros if len(r["texto"]) > 500]
    assert len(longas) == 5, longas

    with open("manifestacoes.json", "w", encoding="utf-8") as f:
        json.dump(registros, f, ensure_ascii=False, indent=2)
    with open("duplicatas_gabarito.json", "w", encoding="utf-8") as f:
        json.dump({"pares": DUPLICATAS}, f, ensure_ascii=False, indent=2)

    from collections import Counter
    print("categorias:", Counter(r["categoria_oficial"] for r in registros))
    print("longas (>500):", longas)
    print("pares duplicados:", len(DUPLICATAS), "->", len(DUPLICATAS), "das 40 manifestações são duplicatas de outra (15%)")
