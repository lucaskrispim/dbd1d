! Second synthetic module for the tests of the Fortran reader (test_caso.py), written
! by the authors: 16 reactions with constant rates, where the tests replace rate 15
! and add lines after TEFF to check how each Fortran construct is evaluated. It
! has the blocks the reader of quimica_zdp.py parses, like sintetico_zdp.F90.
module taxas
  data species_charge(1:species_max) &
  /-1, 0, 0, 1/
  data species_name(1:species_max) &
  /"E   ","A   ","B   ","A^+ "/
  data reaction_sign(1:reactions_max) &
  /"bolsig:A->A^+","A+B=>B+B","A+B=>B+B","A+B=>B+B","A+B=>B+B","A+B=>B+B","A+B=>B+B","A+B=>B+B","A+B=>B+B","A+B=>B+B","A+B=>B+B","A+B=>B+B","A+B=>B+B","A+B=>B+B","A+B=>B+B","A+B=>B+B"/
contains
subroutine ZDPlasKin_reac_rates(Time)
  implicit none
  double precision, intent(in) :: Time
  double precision :: Tgas
  double precision :: EN
  double precision :: Te
  DOUBLE PRECISION :: TEFF
  call ZDPlasKin_bolsig_rates()
  Tgas = ZDPlasKin_cfg(1)
  EN  = ZDPlasKin_cfg(3)
  Te  = ZDPlasKin_cfg(4)
  TEFF = TGAS + 0.12D0 * EN**2
  rrt(01) = bolsig_rates(bolsig_pointer(1))
  rrt(02) = 2.0D-12
  rrt(03) = 3.0D-12
  rrt(04) = 4.0D-12
  rrt(05) = 5.0D-12
  rrt(06) = 6.0D-12
  rrt(07) = 7.0D-12
  rrt(08) = 8.0D-12
  rrt(09) = 9.0D-12
  rrt(10) = 10.0D-12
  rrt(11) = 11.0D-12
  rrt(12) = 12.0D-12
  rrt(13) = 13.0D-12
  rrt(14) = 14.0D-12
  rrt(15) = 1.0D-29
  rrt(16) = 7.0D-11
end subroutine ZDPlasKin_reac_rates
subroutine ZDPlasKin_fex(neq,t,y,ydot)
  call ZDPlasKin_reac_rates(t)
  rrt(01) = rrt(01) * density(01) * density(02)
  rrt(02) = rrt(02) * density(02) * density(03)
  rrt(03) = rrt(03) * density(02) * density(03)
  rrt(04) = rrt(04) * density(02) * density(03)
  rrt(05) = rrt(05) * density(02) * density(03)
  rrt(06) = rrt(06) * density(02) * density(03)
  rrt(07) = rrt(07) * density(02) * density(03)
  rrt(08) = rrt(08) * density(02) * density(03)
  rrt(09) = rrt(09) * density(02) * density(03)
  rrt(10) = rrt(10) * density(02) * density(03)
  rrt(11) = rrt(11) * density(02) * density(03)
  rrt(12) = rrt(12) * density(02) * density(03)
  rrt(13) = rrt(13) * density(02) * density(03)
  rrt(14) = rrt(14) * density(02) * density(03)
  rrt(15) = rrt(15) * density(02) * density(03)
  rrt(16) = rrt(16) * density(02) * density(03)
end subroutine ZDPlasKin_fex
subroutine ZDPlasKin_reac_source_matrix(reac_rate_local,reac_source_local)
  reac_source_local(:,:) = 0.0d0
  reac_source_local(01,01) = + reac_rate_local(01)
  reac_source_local(02,01) = - reac_rate_local(01)
  reac_source_local(04,01) = + reac_rate_local(01)
  reac_source_local(02,02) = - reac_rate_local(02)
  reac_source_local(03,02) = + reac_rate_local(02)
  reac_source_local(02,03) = - reac_rate_local(03)
  reac_source_local(03,03) = + reac_rate_local(03)
  reac_source_local(02,04) = - reac_rate_local(04)
  reac_source_local(03,04) = + reac_rate_local(04)
  reac_source_local(02,05) = - reac_rate_local(05)
  reac_source_local(03,05) = + reac_rate_local(05)
  reac_source_local(02,06) = - reac_rate_local(06)
  reac_source_local(03,06) = + reac_rate_local(06)
  reac_source_local(02,07) = - reac_rate_local(07)
  reac_source_local(03,07) = + reac_rate_local(07)
  reac_source_local(02,08) = - reac_rate_local(08)
  reac_source_local(03,08) = + reac_rate_local(08)
  reac_source_local(02,09) = - reac_rate_local(09)
  reac_source_local(03,09) = + reac_rate_local(09)
  reac_source_local(02,10) = - reac_rate_local(10)
  reac_source_local(03,10) = + reac_rate_local(10)
  reac_source_local(02,11) = - reac_rate_local(11)
  reac_source_local(03,11) = + reac_rate_local(11)
  reac_source_local(02,12) = - reac_rate_local(12)
  reac_source_local(03,12) = + reac_rate_local(12)
  reac_source_local(02,13) = - reac_rate_local(13)
  reac_source_local(03,13) = + reac_rate_local(13)
  reac_source_local(02,14) = - reac_rate_local(14)
  reac_source_local(03,14) = + reac_rate_local(14)
  reac_source_local(02,15) = - reac_rate_local(15)
  reac_source_local(03,15) = + reac_rate_local(15)
  reac_source_local(02,16) = - reac_rate_local(16)
  reac_source_local(03,16) = + reac_rate_local(16)
end subroutine ZDPlasKin_reac_source_matrix
end module taxas
