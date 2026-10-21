! SPDX-License-Identifier: GPL-3.0-or-later
! Generates tabela_energia.dat: the electron coefficients of BOLSIG+ (through
! ZDPlasKin) for the local mean energy approximation (electron energy equation),
! from 1e-4 to 5000 Td (100 points per decade), gas at 300 K, electrons and
! excited states negligible. The simulation interpolates in the mean energy; E/N
! is in the first column for reference.
! Columns: E/N (Td), mean energy (eV), mu*N (1/(cm V s)), D*N (1/(cm s)),
! mu_eps*N (1/(cm V s)), D_eps*N (1/(cm s)), power delivered by the field per N
! (eV cm3/s; in the temporal growth model of BOLSIG+ it is the collisional loss
! plus k_ion*eps, the energy of the created electrons), elastic power and
! inelastic power per N (eV cm3/s; their sum is the collisional loss), collision
! frequency per N (cm3/s) and the electron impact rate coefficients (cm3/s), in
! the order of bolsig_pointer.
! Also writes tabela_eedf.dat, the EEDF of BOLSIG+ (energy in eV, f in eV^-3/2,
! normalized with the integral of sqrt(u) f du = 1) at every node below 0.1 Td
! (Ramsauer region, where the shape changes quickly) and at one in ten above (10
! per decade): one block per node, with the line E/N and mean energy and the
! lines u and f, separated by an empty line. The EEDF is requested after the
! coefficients, so as not to change the order of the BOLSIG+ sweep (which starts
! from the previous solution).
! Also writes tabela_energia_tg.dat, the same columns with the gas temperature
! in front, in blocks of T_g (nodes in t_nos, from 300 to 1500 K, or those of the
! optional file t_nos.txt of the case; the first is the sweep of
! tabela_energia.dat, done before any other so as not to change it), and
! tabela_energia_tg_conferencia.dat, blocks at T_g in the middle of intervals
! (425, 650, 1125 and 1375 K) and one at 300 K with E/N half a step from the
! nodes, only to test the interpolation. The gas density stays that of 300 K: the
! written coefficients are reduced by N.
! The top of E/N is 5000 Td, or the value (Td) of the optional file en_max.txt of
! the case: with N2:H2 25:75 BOLSIG+ crashes with a segmentation fault near 4300 Td.
! The gas composition comes from composicao.txt in the working directory, one line
! per species: NAME FRACTION (fraction of the gas, at 1 atm and 300 K) or NAME
! traco (1 cm-3, as the electrons); without the file, pure argon with Ar+ as trace.
! The number of rate coefficients is that of the BOLSIG+ processes of the case
! (size(bolsig_pointer)), and the header carries the names of the bolsig:
! reactions of kinet.inp, in the same order.
! With calor_rapido.txt (PROCESS_NAME THRESHOLD_EV per line, written by
! ferramentas/caso.py heat), writes tabela_calor_tg.dat: T_g, E/N, mean energy and
! the power per N of these processes, at the T_g nodes: the raw one, sum of
! k*threshold (eV cm3/s), times (1 - (3/2) k T_g / mean energy) (the balance with
! de-excitation, which vanishes for thermal electrons, as in the elastic loss) and
! limited by the total inelastic loss of BOLSIG+ (the gas does not receive more
! than the electrons lose).
! Run with ferramentas/caso.py tables (stdin closed: a failure of BOLSIG+ aborts).
! Before each call to BOLSIG+, the point (T_g, E/N) goes to progresso_tabelas.txt:
! after a crash, ferramentas/caso.py knows where it happened.
program gera_tabela_energia
    use ZDPlasKin
    implicit none
    !
    real(kind=8),parameter :: e_n_min=1.0d-4, ev_por_k=1.380649d-23/1.602176634d-19
    real(kind=8) :: e_n_max=5.0d3
    real(kind=8),allocatable :: t_nos(:)
    integer,parameter :: por_decada=100
    real(kind=8),parameter :: n_ar=101325.0d0/(1.380649d-23*300.0d0)*1.0d-6   ! cm-3, gas at 1 atm and 300 K
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
    write(ue,'(a,i0,a)')"# EEDF of BOLSIG+ (ZDPlasKin), gas at ",nint(t_nos(1)), &
        " K, one block per node: E/N (Td) and mean energy (eV);"
    write(ue,'(a)')"# then energy u (eV) and f (eV^-3/2), normalized with the integral of sqrt(u) f du = 1"
    write(u,'(a,i0,a)')"# electron coefficients of BOLSIG+ (ZDPlasKin) versus mean energy, gas at ",nint(t_nos(1))," K"
    write(u,'(a)')"# E/N (Td), mean energy (eV), mu*N, D*N, mu_eps*N, D_eps*N, P/N, Pel/N, Pinel/N, nu/N"// &
                  trim(nomes)
    open(newunit=ut,file='tabela_energia_tg.dat',status='replace',action='write')
    open(newunit=uc,file='tabela_energia_tg_conferencia.dat',status='replace',action='write')
    write(ut,'(a)')"# electron coefficients of BOLSIG+ (ZDPlasKin) versus gas temperature and mean energy;"
    write(ut,'(a)')"# T_g (K) and then the columns of tabela_energia.dat; one block per T_g"
    write(uc,'(a)')"# as tabela_energia_tg.dat, at T_g outside the nodes, to check the interpolation"
    if(nr > 0)then
        open(newunit=ur,file='tabela_calor_tg.dat',status='replace',action='write')
        write(ur,'(a)')"# fast heating: T_g (K), E/N (Td), mean energy (eV), power per N of the processes of "// &
                       "calor_rapido.txt (eV cm3/s): sum of k*threshold times (1 - Tg/Te), limited by Pinel/N"
    end if
    do it=1,size(t_nos)+size(t_conf)
    if(it <= size(t_nos))then
        tg=t_nos(it)
    else
        tg=t_conf(it-size(t_nos))
    end if
    do i=1,m
        e_n=e_n_min*10.0d0**(real(i-1,kind=8)/real(por_decada,kind=8))
        ! the check block at 300 K lies half a step from the nodes in E/N: it
        ! measures the error of the interpolation in mean energy
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
        ! current point, for the case tool to know where BOLSIG+ crashed
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
        ! mean energy = (3/2) k Te; D of ZDPlasKin comes divided by the density
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
        ! calor_rapido.txt: NAME THRESHOLD per line; without the file, no process
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
        ! index of each process among the collisions of BOLSIG+ (after the first call)
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
        ! t_nos.txt: T_g nodes (K), one per line, increasing; without the file, 300 to 1500 K.
        ! The first node is that of the sweep written in tabela_energia.dat.
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
        if(size(t_nos) < 2)stop "t_nos.txt: at least two T_g nodes"
        if(any(t_nos(2:) <= t_nos(:size(t_nos)-1)))stop "t_nos.txt: T_g nodes must increase"
    end subroutine le_t_nos

    subroutine le_en_max()
        ! en_max.txt: the top of E/N of the table, in Td (without the file, 5000)
        integer :: uf
        logical :: existe
        inquire(file='en_max.txt',exist=existe)
        if(.not. existe)return
        open(newunit=uf,file='en_max.txt',status='old',action='read')
        read(uf,*)e_n_max
        close(uf)
        if(e_n_max <= 1.0d0)stop "en_max.txt: the top of E/N must exceed 1 Td"
    end subroutine le_en_max

    subroutine le_composicao()
        ! composicao.txt: NAME FRACTION or NAME traco; without the file, Ar with Ar+ as trace.
        ! The sum of the fractions is not renormalized: the gas is n_ar times the sum.
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
            ! two fields separated by a space (names may contain commas)
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
        if(n == 0)stop "composicao.txt without species"
    end subroutine le_composicao
end program gera_tabela_energia
