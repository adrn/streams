;--------------------------------------------------------------------
;+
; NAME:
;      ldss_gaincalc
; 
; PURPOSE:
;     Determine relative gain factors from bright flats.
;
; CALLING SEQUENCE:
;     ldss_gaincalc, struc
;
; INPUTS:
;     strct  -- ldss structure, 
;     nccd4  -- set this keyword if in 4-amp mode
;
; OUTPUTS:
;     outputs four chip gains to screen
;
;
; MODIFICATION HISTORY:
;      mg 3/05
;- 
;--------------------------------------------------------------------
Pro ldss_gaincalc, strct, PLOT=plot, nccd4 = nccd4


   qflat = where(strct.type eq 'GFT',nflt)
   print,nflt

   nccd = 2
   if (keyword_set(nccd4)) then nccd = 4 

   gain = fltarr(nccd,nflt)

   for jj=0,nflt-1 do begin

       ; READ ALL FOUR CHIPS
        chip = fltarr(nccd)
        for i=0,nccd-1 do begin
       
           indx = qflat[jj]
           zr='000'
           if (strct[indx].ccdnum gt 9) then zr = '00'
           if (strct[indx].ccdnum gt 99) then zr = '0'
           if (strct[indx].ccdnum gt 999) then zr = ''
           root= 'rawdata/ccd'+ zr+string(strct[indx].ccdnum)
           frame= root + 'c' + string(i+1) + '.fits.gz'
           frame = strcompress(frame,/remove_all)
           
           print,frame
           im = mrdfits(frame)
 
      ; BIAS SUBTRACT
        biasfile = 'Bias/biasc'+string(i+1)+'.fits.gz'
        biasfile = strcompress(biasfile,/remove_all)
        bias = mrdfits(biasfile)
        im = im - bias

           print,biasfile
         ; OVERSCAN SUBTRACT
           print,'Overscan subtracting...'
           ov = djs_median(im[2032:2159,*],1)
           for j = 0,2031 do im[j,*] = im[j,*]-ov
           atv,im


         ; DETERMINE MEDIAN IN EACH CHIP 
           chip[i] = djs_median(im[1100:1800,1400:1900])
           print,chip[i]


        endfor
        gain[1,jj] = chip[1]/chip[0]
        if (keyword_set(nccd4)) then begin
            gain[2,jj] = chip[2]/chip[0]
            gain[3,jj] = chip[3]/chip[0]   
        endif
   endfor



   print, 'chip 1  = 1.0'
   print, 'chip 2  = ',gain[1,*]
   if (keyword_set(nccd4)) then begin
       print, 'chip 3  = ',gain[2,*]
       print, 'chip 4  = ',gain[3,*]
   endif

   if keyword_set(plot) then begin
      im = strcompress(root,/remove_all)
      if keyword_set(nccd4) then ldss_stitch4,im,imout,gain = [1,g2,g3,g4] $
         else ldss_stitch2,im,imout;,gain = [1,g2]
      atv,imout
   endif

end
