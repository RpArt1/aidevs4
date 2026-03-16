classify each description with tags: none, one or more based on job description which is in polish language 

avaliable tags inf format: label: description 
- IT : Obejmuje tworzenie oprogramowania, zarządzanie systemami komputerowymi, infrastrukturą sieciową oraz techniczne wsparcie użytkowników w obszarze nowych technologii.
- transport : Koncentruje się na logistyce, planowaniu przemieszczania dóbr i osób oraz zarządzaniu łańcuchem dostaw w skali lokalnej i międzynarodowej.
- edukacja : Dotyczy procesów dydaktycznych, nauczania, prowadzenia szkoleń oraz wspierania rozwoju kompetencji i zdobywania wiedzy na różnych szczeblach.
- medycyna : Obejmuje opiekę zdrowotną, diagnozowanie i leczenie pacjentów, profilaktykę medyczną oraz pracę w sektorze farmaceutycznym lub terapeutycznym.
- praca z ludźmi : Wymaga wysokich kompetencji interpersonalnych, bezpośredniej obsługi klienta, mediacji, opieki społecznej lub budowania długofalowych relacji.
- praca z pojazdami : Polega na prowadzeniu, operowaniu, konserwacji lub naprawie maszyn i środków transportu lądowego, wodnego lub powietrznego.
- praca fizyczna : Wymaga zaangażowania siły mięśni, sprawności ruchowej, obsługi narzędzi ręcznych lub wykonywania zadań produkcyjnych, montażowych i budowlanych.
 
Think step by step, then respond with label. 


<examples>
    <input-job-descritpion>
        Ten profesjonalista to prawdziwy magik, jeśli chodzi o mechanikę samochodową. Potrafi zdiagnozować każde zadymienie, stukanie czy brak mocy. Jego ręce i wiedza sprawiają, że samochody znów jeżdżą jak nowe.
    </input-job-description>
    <output-label>  
        praca z pojazdami, praca fizyczna
    </output-label>
    <input-job-descritpion>
        Głównym celem tej pracy jest przygotowanie podopiecznych do aktywnego uczestnictwa w życiu społecznym i zawodowym. Przekazuje im nie tylko wiedzę teoretyczną, ale także ucząc praktycznych umiejętności i wartości. Pomaga odkrywać talenty i rozwijać pasje.
    </input-job-description>
    <output-label>
        edukacja, praca z ludźmi
    </output-label>
</examples>


<input-format> 

you will receive a JSON array of job descriptions. Classify each one independently.

[
    {"id": 335, "job": "some job description..."},
    {"id": 336, "job": "another job description..."}
]
</input-format>

