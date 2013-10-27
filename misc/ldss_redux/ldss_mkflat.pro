;--------------------------------------------------------------------
;+
; NAME:
;      ldss_mkflat
; 
; PURPOSE:
;     Make flat field for given science target
;
; CALLING SEQUENCE:
;     ldss_mkflat, strct, objnum
;
; INPUTS:
;     strct  -- B&C structure, 
;     objnum -- object number defined in setcrds 
;
; OUTPUTS:
;     writes normalized file to Flat/fbias.fits
;
;
; MODIFICATION HISTORY:
;     mg 3/05
;- 
;--------------------------------------------------------------------
Pro ldss_mkflat, strct, objnum, mask , nflat,PLOT=plot


   flatroot = 'Flats/flat_'
   flatname = flatroot + string(objnum) + '.fits'
   flatname = strcompress(flatname,/remove_all)

   qflat = where(strct.type eq 'FLT' and strct.objnum eq objnum,nflt)

   print,'Combining '+string(nflt)+' flat frames'
   allflat = fltarr(4064,4064,nflt)

 ; CASE OF SINGLE FLAT FIELD
   if (nflt eq 1) then begin
       print,'WARNING!  single flat field available'
         if keyword_set(nccd4) then $
           ldss_stitch4, strct, qflat[0], flat else $
           ldss_stitch2, strct, qflat[0], flat

       goto, sflt
   endif

 ; CASE OF MULTIPLE FLATS -- NOTE THAT ONLY TWO WILL BE USED!!
   for i=0,nflt-1 do begin
       
       indx = qflat[i]
       if keyword_set(nccd4) then $
          ldss_stitch4, strct, indx, imfinal else $
          ldss_stitch2, strct, indx, imfinal
       allflat[*,*,i] = imfinal

   endfor

 ; CREATE FLAT FRAME from median of many flats 
   print,'Combining flatfield frames'
   flat = x_addtwoflats(allflat[*,*,0],allflat[*,*,1], GAIN=0.7, RN=3.5)


 ; DISPLAY FLAT
   sflt:a=1
   atv,flat
   

; NORMALIZE FLAT FIELD -- could do this as a bspline, for the
;    moment, normalize using a polynomial fit
   s=size(flat)
   naxis1 = s(1)
   naxis2 = s(2)
   nfit = fltarr(naxis1,naxis2)
   xpix = findgen(naxis1)

   cflat = total(flat[*,1400:2500],2)/1101.
   plot,xpix,cflat, xrange=[0,4050],xstyle=1,title='Normalizing flat'
   nfit = x_fitrej(xpix, cflat, 'LEGEND', 13, LSIGMA=5., HSIGMA=5.)
   oplot,xpix,nfit

   tmp = fltarr(naxis1,naxis2)
   for i = 0, naxis2 -1 do begin
       tmp[*,i]  = nfit
   endfor

   nflat = flat / tmp
   atv,nflat

   mwrfits, nflat, flatname, /create



end
