;--------------------------------------------------------------------
;+
; NAME:
;   ldss_combspec
; 
; PURPOSE:
;   Combine all 2D spectrum for a given object.  
;   Allows small spatial shifts (<20pix)
;
; CALLING SEQUENCE:
;   ldss_combspec,strct,obj
;
; INPUTS:
;
;
; OPTIONAL INPUTS:
;
;
; OUTPUTS:
;   writes combined output file to Final/comb_obj**.fits.
;
; OPTIONAL OUTPUTS:
;
;
; COMMENTS:
;
;
; MODIFICATION HISTORY:
;- 
;--------------------------------------------------------------------
Pro ldss_combspec, strct ,objnum, comb, im1,im2
 
 
  if  N_params() Lt 2  then begin
      print,'Syntax - ' + $
        'ldss_combspec, strct, objnum'
      return
  endif


 ; FIND OBJECT FILES
   qobj = where(strct.type eq 'OBJ' and strct.objnum eq objnum,nobj)
   if (nobj eq 0) then begin
       print, 'No reduced images for this object!'
       return
   endif



 ; FINAL COMBINE IMAGE NAME
   combfile = 'Final/comb_' + string(objnum) + '.fits'
   combfile = strcompress(combfile,/remove_all)
       
   zr='000'
   if (strct[qobj[0]].ccdnum gt 9) then zr = '00'
   if (strct[qobj[0]].ccdnum gt 99) then zr = '0'
   if (strct[qobj[0]].ccdnum gt 999) then zr = ''
   ffile = 'Final/f_ccd'+zr +string(strct[qobj[0]].ccdnum)+'.fits'
   ffile = strcompress(ffile,/remove_all)



 ; CASE OF SINGLE REDUCED IMAGE
   print,ffile
   im1   = mrdfits(ffile,1, hdr)
   ivar1 = mrdfits(ffile,2)
   wav   = mrdfits(ffile,3)
    if (nobj eq 1) then begin
       spawn,'cp '+ffile+ ' '+combfile
       return
    endif

   
 ; CASE OF MULTIPLE IMAGES
   comb1 = (im1*ivar1^2) 
   comb2 = (ivar1^2) 
   for i = 1, nobj-1 do begin
         zr='000'
         if (strct[qobj[i]].ccdnum gt 9) then zr = '00'
         if (strct[qobj[i]].ccdnum gt 99) then zr = '0'
         if (strct[qobj[i]].ccdnum gt 999) then zr = ''
         ffile = 'Final/f_ccd'+zr +string(strct[qobj[i]].ccdnum)+'.fits'
         ffile = strcompress(ffile,/remove_all)
         
         print,ffile
         im2   = mrdfits(ffile,1, hdr)
         ivar2 = mrdfits(ffile,2)

       ; CHECK IF IMAGES NEED TO BE REGISTERED-- near halpha
         z=im1 - im2
         ;atv,z[3300:3700,300:600]
;         shft = CROSS_CORR2(im1[3300:3700,300:600], $
;                            im2[3300:3700,300:600], 20, /report)
         
;         if (shft[0]+shft[1] ne 0) then begin
;             print,shft
;             print,'image requires shifting!'
;             stop
;         endif


         comb1 = comb1 + (im2*ivar2^2)
         comb2 = comb2 + (ivar2^2)
  endfor

; WEIGHTED AVERAGE OF IMAGES/NOISE
  comb = comb1/comb2
  combivar = sqrt(comb2)
  q=where(not float(finite(comb)))
  comb[q]=0

  atv,comb;,/align


 ; WRITE FINAL COMBINED IMAGE
   mwrfits, 0, combfile, /create
   mwrfits, comb, combfile, hdr
   mwrfits, combivar, combfile
   mwrfits, wav, combfile
 

end






