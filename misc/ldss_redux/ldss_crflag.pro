;-------------------------------------------------------------
;+
; NAME:
;       SMOOTH2
; PURPOSE:
;       Do multiple smoothing. Gives near Gaussian smoothing.
; CATEGORY:
; CALLING SEQUENCE:
;       b = smooth2(a, w)
; INPUTS:
;       a = array to smooth (1,2, or 3-d).  in
;       w = smoothing window size.          in
; KEYWORD PARAMETERS:
; OUTPUTS:
;       b = smoothed array.                 out
; COMMON BLOCKS:
; NOTES:
; MODIFICATION HISTORY:
;       R. Sterner.  8 Jan, 1987.
;       Johns Hopkins University Applied Physics Laboratory.
;       RES  14 Jan, 1987 --- made both 2-d and 1-d.
;       RES 30 Aug, 1989 --- converted to SUN.
;       R. Sterner, 1994 Feb 22 --- cleaned up some.
;-
;-------------------------------------------------------------
	function smooth2, i, w, help=hlp
 
	if (n_params(0) lt 2) or keyword_set(hlp)  then begin
	  print,' Do multiple smoothing. Gives near Gaussian smoothing.'
	  print,' b = smooth2(a, w)'
	  print,'   a = array to smooth (1,2, or 3-d).  in'
	  print,'   w = smoothing window size.          in'
	  print,'   b = smoothed array.                 out'
	  return, -1
	endif
 
	w1 = w > 2
	w2 = w/2 > 2
 
	i2 = smooth(i, w1)
	i2 = smooth(i2, w1)
	i2 = smooth(i2, w2)
	i2 = smooth(i2, w2)
 
	return, i2
	end

;--------------------------------------------------------------------
;+
; NAME:
;     ldss_crflag
; 
; PURPOSE:
;   Flag cosmic rays.  
;   Return location of CR pixels in the array of indices named crflag. 
;
; CALLING SEQUENCE:
;    ldss_crflag, img, ivar, crflg
;
; INPUTS:
;   img   -- 2D processed image
;   ivar  -- 2D inverse variance
;
; OPTIONAL INPUTS
;   nsig  -- sigma over background to reject
;   psfsig-- size of PSF in pixels
;   smth  -- smoothing length to grow CR hits
;
;   fixchip2 -- mask extra pixels near CR in chip2 to help with
;                    charge transfer problem
;
; OUTPUTS:
;   crflg  -- indx of cr hits
;
; COMMENTS
;   fixed flux levels for fixchip2, add param ytrim
;
; MODIFICATION HISTORY:
;  mg 3/05
;- 
;--------------------------------------------------------------------
Pro ldss_crflag, im, ivar, crflag, nsig=nsig, psfsig=psfsig, smth=smth,$
                 thresh=thresh,fixchip2=fixchip2


  if  N_params() Lt 2  then begin
      print,'Syntax - ' + $
        'ldss_crflag, img, ivar, crflag'
      return
  endif  

  if NOT keyword_set(nsig)   then nsig = 9
  if NOT keyword_set(psfsig) then psfsig = 4
  if NOT keyword_set(smth)   then smth = 3
  if NOT keyword_set(thresh)   then thresh = 0.015

; FLAG OBJECTS WHICH ARE MUCH SHARPER THAN PSF
  psfvals=exp(-0.5*([1.,1.4]/(psfsig))^2)
  reject_cr, im, ivar, psfvals, crrej,niter=8,nsig=nsig
 

; GROW CR HITS BY SMOOTHING MASK IMAGE
  tmp = im
  tmp[*,*] = 0
  tmp[crrej] = 1


  smim = smooth2(tmp,smth)
  crflag = where(smim gt thresh, ncr)

; FIX LDSS3 chip2 charge transfer problem
  if keyword_set(fixchip2) then begin
      s = SIZE(im)
      ncol = s(1)
      col = crflag MOD ncol
      row = crflag / ncol

    ; ADD MASKED COULMNS, DEPENDING ON FLUX LEVEL
      for i = 0,ncr -1 do begin
      
          if (im[crflag[i]] ge 800 and col[i] ge 2031 and $ 
                                  row[i] le 2031-1300) then begin
                 newcol = (col[i] - findgen(10))
                 if (im[crflag[i]] ge 900 and col[i] ge 2031 and $
                                   row[i] le 2031-1300) then begin
                                      newcol = (col[i] - findgen(15))
                 endif
                 smim[newcol,row[i]] = 1 

          endif
      endfor
      crflag = where(smim gt 0.015, ncr)
 endif



 return
end




