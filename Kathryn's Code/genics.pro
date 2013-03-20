; genics generates random selections
; genics_david generates selection  to mimic david's stars
; genics_obs generates selection  to mimic observed data sets

; generates initial conditions for Sgr and debris observations under different
; assumptions
PRO genics
COMMON coords,nstars,x,y,z,vx,vy,vz
rsun=8.
nstars=50

; units
mu=2.5e8
ru=0.63
gee=6.67e-8
msun=1.989e33
cmpkpc=3.085678e21
secperyr=3.1536d7 
tu=((cmpkpc*ru)^1.5)/sqrt(msun)/sqrt(mu*gee)
vu=(ru*cmpkpc*1.e-5)/tu
tu=tu/secperyr/1.e9

; center
READCOL,'../try1/SCFCEN',t,dt,x,y,z,vx,vy,vz
ncen=N_ELEMENTS(x)
xcen=ru*x[ncen-1]
ycen=ru*y[ncen-1]
zcen=ru*z[ncen-1]
vxcen=vu*vx[ncen-1]
vycen=vu*vy[ncen-1]
vzcen=vu*vz[ncen-1]
tnow=tu*t[ncen-1]

; RR Lyraes
READCOL,'../try1/SNAP010',m,x,y,z,vx,vy,vz,s,s,tub,NUMLINE=10000,SKIPLINE=1

; Transform so today is t=0
tub=tu*tub-tnow

iopt=3
; option 1: pick a random sample
IF(iopt EQ 1) THEN p=WHERE(tub LT 0.,count)

; option 2: recent and close debris
IF(iopt EQ 2) THEN BEGIN
   d=ru*SQRT((x+rsun/ru)*(x+rsun/ru)+y*y+z*z)
   p=WHERE((tub LT -0.5) AND (tub GT -3.) AND (d LT 60.),count)
ENDIF

; option 3: even closer debris
IF(iopt EQ 3) THEN BEGIN
   d=ru*SQRT((x+rsun/ru)*(x+rsun/ru)+y*y+z*z)
   p=WHERE(d LT 35.,count)
ENDIF

ipick=FIX(RANDOMU(iseed,nstars)*FLOAT(count))
p=p[ipick]

x=x[p]*ru
y=y[p]*ru
z=z[p]*ru
vx=vx[p]*vu
vy=vy[p]*vu
vz=vz[p]*vu

; add errors?
adderrors

; output
OPENW,4,'ics.dat'
PRINTF,4,xcen,ycen,zcen,vxcen,vycen,vzcen,FORMAT='(6e12.4)'
FOR i=0,nstars-1 DO PRINTF,4,x[i],y[i],z[i],vx[i],vy[i],vz[i],FORMAT='(6e12.4)'
CLOSE, 4

END
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
PRO genics_david
; picks stars form my sim in same regions as David's

COMMON coords,nstars,x,y,z,vx,vy,vz

DEVICE,DECOMPOSED=0
WINDOW,XSIZE=500,YSIZE=500
rsun=8.
nstars=20

; units
mu=2.5e8
ru=0.63
gee=6.67e-8
msun=1.989e33
cmpkpc=3.085678e21
secperyr=3.1536d7 
tu=((cmpkpc*ru)^1.5)/sqrt(msun)/sqrt(mu*gee)
vu=(ru*cmpkpc*1.e-5)/tu
tu=tu/secperyr/1.e9

; center
READCOL,'../try1/SCFCEN',t,dt,x,y,z,vx,vy,vz
ncen=N_ELEMENTS(x)
xcen=ru*x[ncen-1]
ycen=ru*y[ncen-1]
zcen=ru*z[ncen-1]
vxcen=vu*vx[ncen-1]
vycen=vu*vy[ncen-1]
vzcen=vu*vz[ncen-1]
tnow=tu*t[ncen-1]

; RR Lyraes
READCOL,'../try1/SNAP010',m,x,y,z,vx,vy,vz,s,s,tub,NUMLINE=10000,SKIPLINE=1
x=x*ru
y=y*ru
z=z*ru
vx=vx*vu
vy=vy*vu
vz=vz*vu

d=SQRT((x+8.)*(x+8.)+y*y+z*z)
p=WHERE(d LT 35.)
color
PLOT,x,z,PSYM=3
OPLOT,x[p],z[p],PSYM=3,color=2
OPLOT,[xcen,1000.],[zcen,1000.],PSYM=4,THICK=5.,SYMSIZE=2.
OPLOT,[-8.,1000.],[0.,1000.],PSYM=7,THICK=5.,SYMSIZE=2.

p=[0,5,10,15]
READCOL,'../try1/david_starlist_distances.dat',s,s,s,s,s,s,xd,yd,zd,s,s,s,s,s,s,vxd,vyd,vzd
xd=-xd
vxd=-vxd
;OPLOT,xd[p],zd[p],PSYM=5,THICK=5.,SYMSIZE=2.

FOR i=1,4 DO BEGIN
   k=(i-1)*5
   r=SQRT((x-xd[k])^2+(y-yd[k])^2+(z-zd[k])^2)
   v=SQRT((vx-vxd[k])^2+(vy-vyd[k])^2+(vz-vzd[k])^2)
   p=WHERE((r LT 5.) AND (v LT 20.),count)
;   p=WHERE((r LT 10.),count)
   print,count,MIN(r),MIN(v)
   IF(count GE 5)THEN BEGIN
      k=p[0:4]
      IF(i EQ 1)THEN BEGIN
         xs=x[k]
         ys=y[k]
         zs=z[k]
         vxs=vx[k]
         vys=vy[k]
         vzs=vz[k]
      ENDIF
      IF(i GT 1)THEN BEGIN
         xs=[xs,x[k]]
         ys=[ys,y[k]]
         zs=[zs,z[k]]
         vxs=[vxs,vx[k]]
         vys=[vys,vy[k]]
         vzs=[vzs,vz[k]]
      ENDIF
   ENDIF
ENDFOR


x=xs
y=ys
z=zs
vx=vxs
vy=vys
vz=vzs
;OPLOT,x,z,PSYM=4,THICK=1.,SYMSIZE=1.,color=2

;adderrors

; output
;OPENW,4,'ics_david.dat'
;PRINTF,4,xcen,ycen,zcen,vxcen,vycen,vzcen,FORMAT='(6e12.4)'
;FOR i=0,nstars-1 DO PRINTF,4,x[i],y[i],z[i],vx[i],vy[i],vz[i],FORMAT='(6e12.4)'
;CLOSE, 4

END
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
PRO genics_obs,iout
; picks stars form my sim in same regions as Vivas et al (2005) QUEST and
; Watkins et al (2009) Stripe 82 observations

COMMON coords,nstars,x,y,z,vx,vy,vz

IF(iout EQ 0)THEN BEGIN
   WINDOW,1,XSIZE=400,YSIZE=400
   DEVICE,DECOMPOSED=0
ENDIF

IF(iout GT 0)THEN BEGIN
    my_device = !d.name
    SET_PLOT, 'PS' 
    fname="sgr.eps"
    DEVICE, filename=fname,/ENCAPSUL,XSIZE=10,YSIZE=10,/COLOR
ENDIF


rsun=8.
nstars=58

; units
mu=2.5e8
ru=0.63
gee=6.67e-8
msun=1.989e33
cmpkpc=3.085678e21
secperyr=3.1536d7 
tu=((cmpkpc*ru)^1.5)/sqrt(msun)/sqrt(mu*gee)
vu=(ru*cmpkpc*1.e-5)/tu
tu=tu/secperyr/1.e9

; center
READCOL,'../try1/SCFCEN',t,dt,x,y,z,vx,vy,vz
ncen=N_ELEMENTS(x)
xcen=ru*x[ncen-1]
ycen=ru*y[ncen-1]
zcen=ru*z[ncen-1]
vxcen=vu*vx[ncen-1]
vycen=vu*vy[ncen-1]
vzcen=vu*vz[ncen-1]
tnow=tu*t[ncen-1]

; RR Lyraes
READCOL,'../try1/SNAP010',m,x,y,z,vx,vy,vz,s,s,tub,NUMLINE=20000,SKIPLINE=1
x=x*ru
y=y*ru
z=z*ru
vx=vx*vu
vy=vy*vu
vz=vz*vu

color
!P.SYMSIZE=0.1
!P.CHARSIZE=1.5
PLOTSYM,0,/FILL
PLOT,x,z,PSYM=8,XTITLE='X (kpc)',YTITLE='Z (kpc)',XRANGE=[-80,80.],YRANGE=[-80.,80.],/XSTYLE,/YSTYLE,SYMSIZE=0.1

;OPLOT,[xcen,1000.],[zcen,1000.],PSYM=4,THICK=5.,SYMSIZE=1.
OPLOT,[-8.,1000.],[0.,1000.],PSYM=5,THICK=5.,SYMSIZE=1.
p=INDGEN(10000)
x=x[p]
y=y[p]
z=z[p]
vx=vx[p]
vy=vy[p]
vz=vz[p]

; Vivas - 85 stars, 50-60kpc
; From L&M 2010 - lambda ~ 270-300, L1
; => pick stars with angular separation 60-90, in North, distance
; 50-60kpc
; angular position relative to Sgr
xsun=x+rsun
d=SQRT(xsun*xsun+y*y+z*z)
xcensun=xcen+rsun
dcen=SQRT((xcensun)*(xcensun)+ycen*ycen+zcen*zcen)
cosl=(xsun*xcensun+y*ycen+z*zcen)/d/dcen
lam=ACOS(cosl)*180./!PI
p=WHERE((lam GT 60.) AND (lam LT 90.) AND (d GT 30.) AND (d LT 65.) AND (z GT 0.),count)
;p=WHERE((lam GT 60.) AND (lam LT 90.),count)
IF(count GT 0)THEN OPLOT,x[p],z[p],PSYM=8,color=2,SYMSIZE=0.2
; = 674
; which to select?
n=5
cost=TOTAL(0.6*(d[p[0:n-1]]/10.)^2)
print,n,cost,MIN(d[p]),MAX(d[p])
q=p[0:n-1]
; how many closer tham this? =55 => 0.08, so ~8 in QUEST
p=WHERE((lam GT 60.) AND (lam LT 90.) AND (d LT 30.) AND (d GT 15.) AND (z GT 0.),count)
; which to select?
n=8
cost=TOTAL(0.6*(d[p[0:n-1]]/10.)^2)
print,n,cost,MIN(d[p]),MAX(d[p])
q=[q,p[0:n-1]]

; Watkins - 55 stars, 25 kpc, lgal~135 - 180
; L&M10 place in T1
lgal=ATAN(y,xsun)*180./!PI
p=WHERE(lgal GT 135. AND lgal LT 180. AND d GT 20. AND d LT 35. AND z LT 0.,count)
IF(count GT 0)THEN OPLOT,x[p],z[p],PSYM=8,color=4,SYMSIZE=0.2
; which to select?
n=20
cost=TOTAL(0.6*(d[p[0:n-1]]/10.)^2)
print,n,cost,MIN(d[p]),MAX(d[p])
q=[q,p[0:n-1]]

; how many closer tham this? = 75 => 0.45, so ~ 25 in 
p=WHERE(lgal GT 135. AND lgal LT 180. AND d LT 20. AND d GT 7. AND z LT 0.,count)
; which to select?
n=25
cost=TOTAL(0.6*(d[p[0:n-1]]/10.)^2)
print,n,cost,MIN(d[p]),MAX(d[p])
q=[q,p[0:n-1]]

n=N_ELEMENTS(q)
cost=TOTAL(0.6*(d[q]/10.)^2)
print,n,cost,MIN(d[p]),MAX(d[p])

OPLOT,x[q],z[q],PSYM=7,color=5,THICK=4.,SYMSIZE=0.3
nstars=n
x=x[q]
y=y[q]
z=z[q]
vx=vx[q]
vy=vy[q]
vz=vz[q]


;adderrors


; output
;OPENW,4,'ics_obs.dat'
;PRINTF,4,xcen,ycen,zcen,vxcen,vycen,vzcen,FORMAT='(6e12.4)'
;FOR i=0,n-1 DO PRINTF,4,x[i],y[i],z[i],vx[i],vy[i],vz[i],FORMAT='(6e12.4)'
;CLOSE, 4
IF(iout GT 0)THEN BEGIN
    DEVICE, /close_file
    SET_PLOT, my_device
ENDIF


END
;;;;;;;;;;;;;;;;;
PRO adderrors
COMMON coords,nstars,x,y,z,vx,vy,vz

;print,x,y,z,vx,vy,vz,FORMAT='(6e12.4)'
; SUN CENTERED VIEW
rsun=8.
; translate to Sun centered view
x=x+rsun
d=SQRT(x*x+y*y+z*z)
vr=(x*vx+y*vy+z*vz)/d
; proper motions in km/s/kpc
rad=SQRT(x*x+y*y)
vrad=(x*vx+y*vy)/rad
mul=(x*vy-y*vx)/rad/d
mub=(-z*vrad+rad*vz)/d/d
; angular position
sinb=z/d
cosb=rad/d
cosl=x/rad
sinl=y/rad
;
; with THANKS to Horace Smith
; V abs mag RR Lyrae's
;  M_v = (0.214 +/- 0.047)([Fe/H] + 1.5) + 0.45+/-0.05
; Benedict et al. (2011 AJ, 142, 187)
; assuming [Fe/H]=-0.5
mabs=0.65
; Johnson/Cousins (V-IC)  
; (V-IC) color=vmic
; 0.1-0.58
; Guldenschuh et al. (2005 PASP 117, 721)
vmic =0.3
V=mabs+5.*ALOG10(d*100.)

;print,d,mul,mub,vr,cosl,cosb,sinl,sinb,FORMAT='(8e12.4)'
; ADD ERRORS
; 2% distance errors
deld=d*0.02
d=d+deld*RANDOMN(iseed,nstars)

; 5 km/s velocity
delv=5.0d0
vr=vr+delv*RANDOMN(iseed,nstars)

; GAIA proper motions
; April 2011:
; http://www.rssd.esa.int/index.php?project=GAIA&page=Science_Performance#chapter1
;        25 muas at V=15 (RR Lyraes are ~A-F type :between B and G star)
;        300 muas at V=20

; GAIA G mag
g = V - 0.0257 - 0.0924*vmic- 0.1623*vmic^2 + 0.0090*vmic^3
;
z =10^(0.4*(g - 15.))
p=WHERE(g LT 12.,count)
IF(count GT 0)THEN z[p] =10^(0.4*(12. - 15.))
; "end of mission parallax standard"
; σπ [μas] = sqrt(9.3 + 658.1 · z + 4.568 · z^2) · [0.986 + (1 - 0.986) · (V-IC)]
dp= SQRT(9.3 + 658.1*z + 4.568*z^2)*(0.986 + (1 - 0.986)*vmic)
; assume 5 year baseline, mas/year
dmu=dp/5.
; too optimistic: following suggests factor 2 more realistic
;http://www.astro.utu.fi/~cflynn/galdyn/lecture10.html 
; - and Sanjib suggests factor 0.526
dmu=0.526*dp

; translate to radians/year
conv1=!PI/180./60./60./1.e6
; translate to km/s from  kpc/year 
kmpkpc=3.085678e16
secperyr=3.1536d7 
conv2=kmpkpc/secperyr
; 
dmu=dmu*conv1*conv2
; 
mul=mul+dmu*RANDOMN(iseed,nstars)
mub=mub+dmu*RANDOMN(iseed,nstars)

; CONVERT BACK
x=d*cosb*cosl-rsun
y=d*cosb*sinl
z=d*sinb
;
vx=vr*cosb*cosl-d*mul*sinl-d*mub*sinb*cosl
vy=vr*cosb*sinl+d*mul*cosl-d*mub*sinb*sinl
vz=vr*sinb+d*mub*cosb
;print,d,mul,mub,vr,cosl,cosb,sinl,sinb,FORMAT='(8e12.4)'
;print,x,y,z,vx,vy,vz,FORMAT='(6e12.4)'

; calculate RR Lyrae cost
cost=TOTAL(0.6*(d/10.)^2)
;avecost=TOTAL(cost)/1000.
print,cost
RETURN
END

;;;;;;;;;;;;;;;;;;;
PRO gaia

; GAIA proper motions
; April 2011:
; http://www.rssd.esa.int/index.php?project=GAIA&page=Science_Performance#chapter1
;        25 muas at V=15 (RR Lyraes are ~A-F type :between B and G star)
;        300 muas at V=20
; out of date:
;       11 muas at V=15, 160 at V=20
;       roughly (?) dmu=0.011 * 10^ V/5
;       (doesn't quite scale as expected?)

; with THANKS to Horace Smith
; V abs mag RR Lyrae's
;  M_v = (0.214 +/- 0.047)([Fe/H] + 1.5) + 0.45+/-0.05
; Benedict et al. (2011 AJ, 142, 187)
; assuming [Fe/H]=-0.5
mabs=0.65
; Johnson/Cousins (V-IC)  
; (V-IC) color=vmic
; 0.1-0.58
; Guldenschuh et al. (2005 PASP 117, 721)
vmic =0.3

; => at d=10/20/30/40/50
d=FINDGEN(6)*20.+20.
V=mabs+5.*ALOG10(d*100.)
print,d
print,V
; GAIA G mag
g = V - 0.0257 - 0.0924*vmic- 0.1623*vmic^2 + 0.0090*vmic^3
;
z =10^(0.4*(g - 15.))
p=WHERE(g LT 12.,count)
IF(count GT 0)THEN z[p] =10^(0.4*(12. - 15.))

; "end of mission parallax standard"
; σπ [μas] = sqrt(9.3 + 658.1 · z + 4.568 · z^2) · [0.986 + (1 - 0.986) · (V-IC)]
; parallax accuracy
;dp=0.011 * 10^(V/5)
dp= SQRT(9.3 + 658.1*z + 4.568*z^2)*(0.986 + (1 - 0.986)*vmic)
print,dp
; assume ~factor 2 conversion to pm
; Sanjib suggests 0.526
dmu=0.526*dp
print,dmu
; translate to radians/year
conv1=!PI/180./60./60./1.e6
; translate to km/s from  kpc/year 
kmpkpc=3.085678e16
secperyr=3.1536d7 
conv2=kmpkpc/secperyr
; 
print,conv1*conv2
dv=dmu*conv1*conv2*d
;
print,dv,FORMAT='(6f10.2)'
END

