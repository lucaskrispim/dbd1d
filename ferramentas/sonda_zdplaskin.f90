! SPDX-License-Identifier: GPL-3.0-or-later
! Probe of the ZDPlasKin module generated for a case (ferramentas/caso.py
! zdplaskin): for each state of estados.txt (T_g in K, E/N in Td and the density of
! each species in cm-3, in the order of species_name), lets ZDPlasKin and BOLSIG+
! compute everything and writes sonda.txt, one line per state: T_g, E/N, T_e (K),
! the total density of neutrals (ALL_NEUTRAL), the BOLSIG+ rates in the order of
! bolsig_pointer, the rate of each reaction (cm-3/s) and the source of each species
! (cm-3/s). The simulator, with the same inputs, must reproduce the rates and the
! sources (prototipo/testes/test_geracao_zdplaskin.py).
! Run in the case folder (BOLSIG+ reads bolsigdb.dat from there), with stdin closed.
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
