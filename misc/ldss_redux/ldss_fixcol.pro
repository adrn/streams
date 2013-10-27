;--------------------------------------------------------------------
;+
; NAME:
;    ldss_fixcol
; 
; PURPOSE:
;    Interpolate over bad columns as defined in ldss_badcol.dat
;
; CALLING SEQUENCE:
;   ldss_fixpix, im 
;
; INPUTS:
;
;
; OPTIONAL INPUTS:
;
;
; OUTPUTS:
;
;
; COMMENTS:
;
;
; MODIFICATION HISTORY:
;    mg 3/05
;- 
;--------------------------------------------------------------------
Pro ldss_fixcol, im, ivar

  if  N_params() LT 1  then begin
      print,'Syntax - ' + $
        'ldss_fixpix, im'
      return
  endif

 ; LDSS3 HeAR ARC LIST
   dodgy_path = getenv('DODGY_DIR')
   badlist = dodgy_path + '/pro/LDSS3/ldss_badcol_aug05.dat'
   readcol,badlist,f='i,i,i,a',x1,x2,y1,y2

   s=size(im)

   for i=0,n_elements(x1)-1 do begin

       if (y2[i] eq '*') then y2[i]=s(2)-1

     ; REPLACE BAD PIXELS WITH AVERAGE ON EITHER SIDE
       replc = djs_median(im[x1[i]-1:x2[i]+1, y1[i]:y2[i]], 1)

       sz = x2[i]-x1[i]
       for j=0,sz do begin
           x = x1[i] +j
           im[x, y1[i]:y2[i]] = replc

         ; SET INVERSE VARIANCE IN BAD COLUMNS = 0
           ivar[x, y1[i]:y2[i]] = 0

       endfor

   endfor


end





