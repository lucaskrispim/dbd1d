! SPDX-License-Identifier: GPL-3.0-or-later
! Sonda do ZDPlasKin gerado para um caso (ferramentas/caso.py zdplaskin): para
! cada estado de estados.txt (T_g em K, E/N em Td e a densidade de cada espécie
! em cm-3, na ordem de species_name), deixa o ZDPlasKin e o BOLSIG+ calcularem
! tudo e grava em sonda.txt, uma linha por estado: T_g, E/N, T_e (K), a
! densidade total de neutros (ALL_NEUTRAL), as taxas do BOLSIG+ na ordem de
! bolsig_pointer, a taxa de cada reação (cm-3/s) e a fonte de cada espécie
! (cm-3/s). O protótipo, com as mesmas entradas, tem de reproduzir as taxas e
! as fontes (prototipo/testes/test_geracao_zdplaskin.py).
! Rodar na pasta do caso (o BOLSIG+ lê o bolsigdb.dat de lá), com stdin fechado.
program sonda_zdplaskin
    use ZDPlasKin
    implicit none
    integer :: u, v, n, k, j
    double precision :: tg, e_n, te, neutros
    double precision :: dens(species_max), fontes(species_max), taxas(reactions_max)
    double precision, allocatable :: kb(:)
    !
    allocate(kb(size(bolsig_pointer)))
    open(newunit=u,file='estados.txt',status='old',action='read')
    open(newunit=v,file='sonda.txt',status='replace',action='write')
    read(u,*) n
    do k=1,n
        read(u,*) tg, e_n, dens
        call ZDPlasKin_reset()
        call ZDPlasKin_set_config(SILENCE_MODE=.true.)
        do j=1,species_max
            call ZDPlasKin_set_density(species_name(j),dens(j))
        end do
        call ZDPlasKin_set_conditions(GAS_TEMPERATURE=tg,REDUCED_FIELD=e_n)
        call ZDPlasKin_bolsig_rates(lbolsig_force=.true.)
        call ZDPlasKin_get_conditions(ELEC_TEMPERATURE=te)
        call ZDPlasKin_get_density_total(ALL_NEUTRAL=neutros)
        do j=1,size(bolsig_pointer)
            kb(j)=bolsig_rates(bolsig_pointer(j))
        end do
        call ZDPlasKin_get_rates(SOURCE_TERMS=fontes,REACTION_RATES=taxas)
        write(v,'(*(es27.17e3))') tg, e_n, te, neutros, kb, taxas, fontes
    end do
    close(u)
    close(v)
end program sonda_zdplaskin
