! Synthetic module for the tests of the positive column (dbd1d/coluna.py), written by
! the authors, in the blocks of a module of the ZDPlasKin preprocessor that
! quimica_zdp.py reads. Background N2 with four electron impact processes, whose
! coefficients come from the synthetic table of test_coluna_sintetica.py: ionization,
! dissociation and excitation of N2(v = 1) and N2(v = 2). No other reaction, so that
! the densities and the sustaining field have closed forms.
module coluna_sintetica
  data species_charge(1:species_max) &
  /-1, 0, 0, 0, 0, 1/
  data species_name(1:species_max) &
  /"E      ","N2     ","N2(v1) ","N2(v2) ","N      ","N2^+   "/
  data reaction_sign(1:reactions_max) &
  /"bolsig:N2->N2^+  ","bolsig:N2->N     ","bolsig:N2->N2(v1)","bolsig:N2->N2(v2)"/
contains
subroutine ZDPlasKin_reac_rates(Time)
  implicit none
  double precision, intent(in) :: Time
  call ZDPlasKin_bolsig_rates()
  rrt(01) = bolsig_rates(bolsig_pointer(1))
  rrt(02) = bolsig_rates(bolsig_pointer(2))
  rrt(03) = bolsig_rates(bolsig_pointer(3))
  rrt(04) = bolsig_rates(bolsig_pointer(4))
end subroutine ZDPlasKin_reac_rates
subroutine ZDPlasKin_fex(neq,t,y,ydot)
  call ZDPlasKin_reac_rates(t)
  rrt(01) = rrt(01) * density(01) * density(02)
  rrt(02) = rrt(02) * density(01) * density(02)
  rrt(03) = rrt(03) * density(01) * density(02)
  rrt(04) = rrt(04) * density(01) * density(02)
end subroutine ZDPlasKin_fex
subroutine ZDPlasKin_reac_source_matrix(reac_rate_local,reac_source_local)
  reac_source_local(:,:) = 0.0d0
  reac_source_local(01,01) = + reac_rate_local(01)
  reac_source_local(02,01) = - reac_rate_local(01)
  reac_source_local(06,01) = + reac_rate_local(01)
  reac_source_local(02,02) = - reac_rate_local(02)
  reac_source_local(05,02) = + reac_rate_local(02) * 2.d0
  reac_source_local(02,03) = - reac_rate_local(03)
  reac_source_local(03,03) = + reac_rate_local(03)
  reac_source_local(02,04) = - reac_rate_local(04)
  reac_source_local(04,04) = + reac_rate_local(04)
end subroutine ZDPlasKin_reac_source_matrix
end module coluna_sintetica
