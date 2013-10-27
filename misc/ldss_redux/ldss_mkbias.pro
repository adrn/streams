;--------------------------------------------------------------------
;+
; NAME:
;      ldss_mkbias
; 
; PURPOSE:
;     Make bias for each of 4 LDSS3 chips.
;     Bias frames defined in LDSS3 structure.  Median combines 
;     all frames with strct.type ='ZRO" 
;
; CALLING SEQUENCE:
;     ldss_mkbias, strct
;
; INPUTS:
;     strct  -- LDSS structure  
;
; OUTPUTS:
;     writes bias file to Bias/bias.fits
;
;
; MODIFICATION HISTORY:
;   mg 3/05
;- 
;--------------------------------------------------------------------
Pro ldss_mkbias, strct, PLOT=plot, nccd4 = nccd4

   nccd = 2
   if (keyword_set(nccd4)) then nccd = 4 


   biasroot = 'Bias/bias'
   qbias = where(strct.type eq 'ZRO',c)

   print,'Median combining '+string(c)+' bias frames'

   tmp = mrdfits(strct[qbias[0]].ccdframe)
   s=size(tmp)
   chipbias = fltarr(s(1),s(2),c)

   
 ; MAKE SEPARATE BIAS FOR EACH CHIP
   for j = 1, nccd do begin

       for i=0,c-1 do begin
       
           indx = qbias[i]
           zr='000'
           if (strct[indx].ccdnum gt 9) then zr = '00'
           if (strct[indx].ccdnum gt 99) then zr = '0'
           if (strct[indx].ccdnum gt 999) then zr = ''
           frame= 'rawdata/ccd'+ zr+string(strct[indx].ccdnum) + $
                 'c' + string(j) + '.fits.gz'
           frame = strcompress(frame,/remove_all)

           print,'Reading frame ',frame
           chipbias[*,*,i] = mrdfits(frame)

         ; CHECK IF SOMETHING IS MESSED UP
           if (strct[indx].exp gt 0) then begin
               print, 'Non-zero exposure times!'
               print,strct[indx].exp
           endif


       endfor

    ; MEDIAN COMBINE INDIVIDUAL BIAS FRAMES 
      bias = djs_median(chipbias,3)

    ; WRITE TO FILE
      biasname = biasroot + 'c' + string(j) + '.fits'
      biasname = strcompress(biasname,/remove_all)

      print,'Writing bias frame ',biasname
      mwrfits, bias, biasname, /create
      
      if keyword_set(plot) then atv,bias

   endfor

end
