! SPDX-License-Identifier: GPL-3.0-or-later
! Gera tabela_energia.dat: os coeficientes dos elétrons do BOLSIG+ (via
! ZDPlasKin) para a aproximação da energia média local (equação de energia dos
! elétrons), de 1e-4 a 5000 Td (100
! pontos por década), gás a 300 K, elétrons e Ar* desprezíveis. A simulação
! interpola pela energia média; E/N fica na primeira coluna para conferência.
! Colunas: E/N (Td), energia média (eV), mu*N (1/(cm V s)), D*N (1/(cm s)),
! mu_eps*N (1/(cm V s)), D_eps*N (1/(cm s)), potência que o campo entrega por N
! (eV cm3/s; no modelo de crescimento temporal do BOLSIG+ ela é a perda por
! colisões mais k_ion*eps, a energia dos elétrons criados), potência elástica e
! potência inelástica por N (eV cm3/s; a soma das duas é a perda por colisões),
! frequência de colisão por N (cm3/s) e as taxas de impacto eletrônico
! (cm3/s), na ordem de bolsig_pointer.
! Grava também tabela_eedf.dat, a EEDF do BOLSIG+ (energia em eV, f em
! eV^-3/2, normalizada com a integral de sqrt(u) f du = 1) em todos os nós
! abaixo de 0,1 Td (região de Ramsauer, onde a forma muda depressa) e em um de
! cada dez acima (10 por década): um bloco por nó, com a linha E/N e energia média e as
! linhas u e f, separados por linha vazia. A EEDF é pedida depois dos
! coeficientes, para não mudar a ordem da varredura do BOLSIG+ (que parte da
! solução anterior).
! Grava ainda tabela_energia_tg.dat, as mesmas colunas com a temperatura do gás
! na frente, em blocos de T_g (nós em t_nos, de 300 a 1500 K, ou os do arquivo opcional
! t_nos.txt do caso; o primeiro é a varredura de
! tabela_energia.dat, feita antes de qualquer outra para não mudá-la), e
! tabela_energia_tg_conferencia.dat, blocos em T_g no meio de intervalos (425, 650,
! 1125 e 1375 K) e um a 300 K
! com E/N a meio passo dos nós, só para testar a interpolação. A densidade do gás fica a de 300 K: os coeficientes gravados
! são reduzidos por N.
! O topo de E/N é 5000 Td, ou o valor (Td) do arquivo opcional en_max.txt do caso:
! com N2:H2 25:75 o BOLSIG+ cai com falha de segmentação perto de 4300 Td.
! A composição do gás vem de composicao.txt no diretório de execução, uma linha
! por espécie: NOME FRACAO (fração do gás, a 1 atm e 300 K) ou NOME traco
! (1 cm-3, como os elétrons); sem o arquivo, argônio puro com Ar+ como traço.
! O número de taxas é o de processos do BOLSIG+ do caso (size(bolsig_pointer)),
! e o cabeçalho traz os nomes das reações bolsig: do kinet.inp, na mesma ordem.
! Com calor_rapido.txt (NOME_DO_PROCESSO LIMIAR_EV por linha, escrito por
! ferramentas/caso.py calor), grava tabela_calor_tg.dat: T_g, E/N, energia média e
! a potência por N desses processos, nos nós de T_g: a bruta, soma de k*limiar
! (eV cm3/s), vezes (1 - (3/2) k T_g / energia média) (o saldo com a desexcitação,
! que zera com os elétrons térmicos, como na perda elástica) e limitada pela perda
! inelástica total do BOLSIG+ (o gás não recebe mais do que os elétrons perdem).
! Rodar com make tabela-energia ou ferramentas/caso.py tabelas (stdin fechado:
! falha do BOLSIG+ aborta). Antes de cada chamada ao BOLSIG+, o ponto (T_g, E/N)
! vai para progresso_tabelas.txt: numa queda, ferramentas/caso.py sabe onde foi.
program gera_tabela_energia
    use ZDPlasKin
    implicit none
    !
    real(kind=8),parameter :: e_n_min=1.0d-4, ev_por_k=1.380649d-23/1.602176634d-19
    real(kind=8) :: e_n_max=5.0d3
    real(kind=8),allocatable :: t_nos(:)
    integer,parameter :: por_decada=100
    real(kind=8),parameter :: n_ar=101325.0d0/(1.380649d-23*300.0d0)*1.0d-6   ! cm-3, gás a 1 atm e 300 K
    real(kind=8),parameter :: t_nos_padrao(15)=[300.0d0,325.0d0,350.0d0,375.0d0,400.0d0,450.0d0,500.0d0,550.0d0, &
                                         600.0d0,700.0d0,800.0d0,900.0d0,1000.0d0,1250.0d0,1500.0d0]
    real(kind=8),parameter :: t_conf(5)=[300.0d0,425.0d0,650.0d0,1125.0d0,1375.0d0]
    real(kind=8) :: e_n, te, mun, d, mueps, deps, pot, pel, pinel, nun, eedf(2,2000), tg
    real(kind=8),allocatable :: k(:), fracao(:)
    character(len=32),allocatable :: especie(:)
    character(len=20000) :: nomes
    integer :: i, j, m, u, ue, ut, uc, it, nk, ur, nr, up
    character(len=64),allocatable :: rap_nome(:)
    real(kind=8),allocatable :: rap_lim(:)
    integer,allocatable :: rap_k(:)
    real(kind=8) :: prap
    !
    call le_composicao()
    call le_calor_rapido()
    call le_en_max()
    call le_t_nos()
    nk=size(bolsig_pointer)
    allocate(k(nk))
    nomes=""
    do i=1,size(reaction_sign)
        if(reaction_sign(i)(1:7) == "bolsig:")nomes=trim(nomes)//", k("//trim(reaction_sign(i)(8:))//")"
    end do
    m=nint(log10(e_n_max/e_n_min)*real(por_decada,kind=8))+1
    open(newunit=u,file='tabela_energia.dat',status='replace',action='write')
    open(newunit=ue,file='tabela_eedf.dat',status='replace',action='write')
    write(ue,'(a,i0,a)')"# EEDF do BOLSIG+ (ZDPlasKin), gas a ",nint(t_nos(1))," K, um bloco por no: E/N (Td) e energia media (eV);"
    write(ue,'(a)')"# depois energia u (eV) e f (eV^-3/2), normalizada com a integral de sqrt(u) f du = 1"
    write(u,'(a,i0,a)')"# coeficientes dos eletrons do BOLSIG+ (ZDPlasKin) por energia media, gas a ",nint(t_nos(1))," K"
    write(u,'(a)')"# E/N (Td), energia media (eV), mu*N, D*N, mu_eps*N, D_eps*N, P/N, Pel/N, Pinel/N, nu/N"// &
                  trim(nomes)
    open(newunit=ut,file='tabela_energia_tg.dat',status='replace',action='write')
    open(newunit=uc,file='tabela_energia_tg_conferencia.dat',status='replace',action='write')
    write(ut,'(a)')"# coeficientes dos eletrons do BOLSIG+ (ZDPlasKin) por temperatura do gas e energia media;"
    write(ut,'(a)')"# T_g (K) e depois as colunas de tabela_energia.dat; um bloco por T_g"
    write(uc,'(a)')"# como tabela_energia_tg.dat, em T_g fora dos nos, para conferir a interpolacao"
    if(nr > 0)then
        open(newunit=ur,file='tabela_calor_tg.dat',status='replace',action='write')
        write(ur,'(a)')"# calor rapido: T_g (K), E/N (Td), energia media (eV), potencia por N dos processos de "// &
                       "calor_rapido.txt (eV cm3/s): soma de k*limiar vezes (1 - Tg/Te), limitada por Pinel/N"
    end if
    do it=1,size(t_nos)+size(t_conf)
    if(it <= size(t_nos))then
        tg=t_nos(it)
    else
        tg=t_conf(it-size(t_nos))
    end if
    do i=1,m
        e_n=e_n_min*10.0d0**(real(i-1,kind=8)/real(por_decada,kind=8))
        ! o bloco de conferência a 300 K fica a meio passo dos nós em E/N: mede
        ! o erro da interpolação em energia média (lacuna V61)
        if(it == size(t_nos)+1)e_n=e_n*10.0d0**(0.5d0/real(por_decada,kind=8))
        call ZDPlasKin_reset()
        call ZDPlasKin_set_config(SILENCE_MODE=.true.)
        do j=1,size(especie)
            if(fracao(j) > 0.0d0)call ZDPlasKin_set_density(trim(especie(j)),n_ar*fracao(j))
        end do
        call ZDPlasKin_set_density('E',1.0d0)
        do j=1,size(especie)
            if(fracao(j) < 0.0d0)call ZDPlasKin_set_density(trim(especie(j)),1.0d0)
        end do
        call ZDPlasKin_set_conditions(GAS_TEMPERATURE=tg,REDUCED_FIELD=e_n)
        ! ponto corrente, para a ferramenta de caso saber onde o BOLSIG+ caiu
        open(newunit=up,file='progresso_tabelas.txt',status='replace',action='write')
        write(up,'(2es24.16)')tg,e_n
        close(up)
        call ZDPlasKin_bolsig_rates(lbolsig_force=.true.)
        call ZDPlasKin_get_conditions(ELEC_TEMPERATURE=te,ELEC_MOBILITY_N=mun,ELEC_DIFF_COEFF=d, &
                                      ELEC_MU_EPS_N=mueps,ELEC_DIFF_EPS_N=deps,ELEC_POWER_N=pot, &
                                      ELEC_POWER_ELASTIC_N=pel,ELEC_POWER_INELASTIC_N=pinel,ELEC_FREQUENCY_N=nun)
        do j=1,nk
            k(j)=bolsig_rates(bolsig_pointer(j))
        end do
        ! energia média = (3/2) k Te; D do ZDPlasKin vem dividido pela densidade
        if(it == 1)write(u,'(*(es24.16))')e_n,1.5d0*te*ev_por_k,mun,d*n_ar,mueps,deps,pot,pel,pinel,nun,k
        if(nr > 0 .and. it <= size(t_nos))then
            if(rap_k(1) == 0)call liga_calor_rapido()
            prap=0.0d0
            do j=1,nr
                prap=prap+bolsig_rates(rap_k(j))*rap_lim(j)
            end do
            prap=max(0.0d0,min(prap*(1.0d0-1.5d0*tg*ev_por_k/(1.5d0*te*ev_por_k)),pinel))
            write(ur,'(*(es24.16))')tg,e_n,1.5d0*te*ev_por_k,prap
        end if
        if(it <= size(t_nos))then
            write(ut,'(*(es24.16))')tg,e_n,1.5d0*te*ev_por_k,mun,d*n_ar,mueps,deps,pot,pel,pinel,nun,k
        else
            write(uc,'(*(es24.16))')tg,e_n,1.5d0*te*ev_por_k,mun,d*n_ar,mueps,deps,pot,pel,pinel,nun,k
        end if
        if(it == 1 .and. (e_n < 0.1d0 .or. mod(i-1,por_decada/10) == 0))then
            call ZDPlasKin_get_conditions(ELEC_EEDF=eedf)
            write(ue,'(a)')""
            write(ue,'(2es24.16)')e_n,1.5d0*te*ev_por_k
            do j=1,size(eedf,dim=2)
                if(eedf(2,j) <= 0.0d0) exit
                write(ue,'(2es24.16)')eedf(1,j),eedf(2,j)
            end do
        end if
    end do
    if(nr > 0 .and. it <= size(t_nos))write(ur,'(a)')""
    if(it <= size(t_nos))then
        write(ut,'(a)')""
    else
        write(uc,'(a)')""
    end if
    end do
    close(u)
    close(ue)
    close(ut)
    close(uc)
    if(nr > 0)close(ur)
    write(*,'(a,i0,a,i0,a)')"tabela_energia.dat: ",m," pontos, ",nk," taxas"
contains
    subroutine le_calor_rapido()
        ! calor_rapido.txt: NOME LIMIAR por linha; sem o arquivo, nenhum processo
        character(len=200) :: linha
        integer :: uf, ios
        logical :: existe
        nr=0
        allocate(rap_nome(0),rap_lim(0))
        inquire(file='calor_rapido.txt',exist=existe)
        if(.not. existe)return
        open(newunit=uf,file='calor_rapido.txt',status='old',action='read')
        do
            read(uf,'(a)',iostat=ios)linha
            if(ios /= 0)exit
            if(len_trim(linha) == 0)cycle
            linha=adjustl(linha)
            rap_nome=[rap_nome,linha(1:64)]
            rap_nome(size(rap_nome))=linha(1:index(linha,' ')-1)
            rap_lim=[rap_lim,0.0d0]
            read(linha(index(linha,' '):),*)rap_lim(size(rap_lim))
        end do
        close(uf)
        nr=size(rap_nome)
        allocate(rap_k(nr))
        rap_k=0
    end subroutine le_calor_rapido

    subroutine liga_calor_rapido()
        ! índice de cada processo entre as colisões do BOLSIG+ (depois da primeira chamada)
        character(len=200) :: nome
        integer :: a, b
        do a=1,nr
            do b=1,size(bolsig_rates)
                call ZDPlasKin_bolsig_GetReactionName(nome,b)
                if(trim(adjustl(nome)) == trim(rap_nome(a)))rap_k(a)=b
            end do
            if(rap_k(a) == 0)then
                write(*,'(a)')"processo de calor_rapido.txt fora do BOLSIG+: "//trim(rap_nome(a))
                stop 1
            end if
        end do
    end subroutine liga_calor_rapido

    subroutine le_t_nos()
        ! t_nos.txt: nós de T_g (K), um por linha, crescentes; sem o arquivo, de 300 a 1500 K.
        ! O primeiro nó é o da varredura gravada em tabela_energia.dat.
        integer :: uf, ios
        real(kind=8) :: x
        logical :: existe
        inquire(file='t_nos.txt',exist=existe)
        if(.not. existe)then
            t_nos=t_nos_padrao
            return
        end if
        allocate(t_nos(0))
        open(newunit=uf,file='t_nos.txt',status='old',action='read')
        do
            read(uf,*,iostat=ios)x
            if(ios /= 0)exit
            t_nos=[t_nos,x]
        end do
        close(uf)
        if(size(t_nos) < 2)stop "t_nos.txt: pelo menos dois nós de T_g"
        if(any(t_nos(2:) <= t_nos(:size(t_nos)-1)))stop "t_nos.txt: nós de T_g precisam ser crescentes"
    end subroutine le_t_nos

    subroutine le_en_max()
        ! en_max.txt: o topo de E/N da tabela, em Td (sem o arquivo, 5000)
        integer :: uf
        logical :: existe
        inquire(file='en_max.txt',exist=existe)
        if(.not. existe)return
        open(newunit=uf,file='en_max.txt',status='old',action='read')
        read(uf,*)e_n_max
        close(uf)
        if(e_n_max <= 1.0d0)stop "en_max.txt: topo de E/N precisa passar de 1 Td"
    end subroutine le_en_max

    subroutine le_composicao()
        ! composicao.txt: NOME FRACAO ou NOME traco; sem o arquivo, Ar com Ar+ de traço.
        ! A soma das frações não é renormalizada: o gás é n_ar vezes a soma.
        character(len=200) :: linha, valor
        integer :: uf, ios, n
        logical :: existe
        inquire(file='composicao.txt',exist=existe)
        if(.not. existe)then
            especie=[character(len=32) :: 'Ar','Ar^+']
            fracao=[1.0d0,-1.0d0]
            return
        end if
        allocate(especie(0),fracao(0))
        open(newunit=uf,file='composicao.txt',status='old',action='read')
        n=0
        do
            read(uf,'(a)',iostat=ios)linha
            if(ios /= 0)exit
            if(len_trim(linha) == 0 .or. linha(1:1) == '#')cycle
            n=n+1
            ! dois campos separados por espaço (nomes podem ter vírgula)
            linha=adjustl(linha)
            nomes=linha(1:index(linha,' ')-1)
            valor=adjustl(linha(index(linha,' '):))
            especie=[especie,nomes(1:32)]
            if(trim(valor) == 'traco')then
                fracao=[fracao,-1.0d0]
            else
                fracao=[fracao,0.0d0]
                read(valor,*)fracao(n)
            end if
        end do
        close(uf)
        if(n == 0)stop "composicao.txt sem especies"
    end subroutine le_composicao
end program gera_tabela_energia
