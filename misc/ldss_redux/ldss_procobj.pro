;--------------------------------------------------------------------
;+
; NAME:
;     ldss_procobj
; 
; PURPOSE:
;    Process single object.  Bias subtract, flat field, cr flag, sky subtract.
;
; CALLING SEQUENCE:
;    ldss, strct, objnum, /plot
;
; INPUTS:
;    strct  -- ldss structure
;    objnum -- object number to be processed (eg. objnum = 2)
;
; OPTIONAL INPUTS:
;
;
; OUTPUTS:
;    imred  -- final reduced image    
;    imsky  -- semi-reduced image (no cr flag or sky subtraction)
;
;    Write reduced image/var/wave to Final/ccd00**.fits
;        0 -- header
;        1 -- image axis
;        2 -- inverse variance
;        3 -- wavelength img
;
; COMMENTS:
;
;
; MODIFICATION HISTORY:
;     mg 3/05
;- 
;--------------------------------------------------------------------
Pro ldss_procobj,strct,objnum, imred, imsky = imsky, plot=plot, nccd4=nccd4


  if  N_params() Lt 1  then begin
      print,'Syntax - ' + $
        'ldss_procobj, strct'
      return
  endif 
  loadct,0
  !p.multi=0

 ; FIND OBJECT FILES
   qobj = where(strct.type eq 'OBJ' and strct.objnum eq objnum,nobj)


 ; PREPARE FLATFIELD IMAGE IF NEEDED
   flatname = 'Flats/flat_' + string(objnum) + '.fits'
   flatname = strcompress(flatname,/remove_all)
   a = file_search(flatname, count = cnt)
   if (cnt eq 0) then begin
      ; IF FLAT IMAGE DOESN'T EXIST, THEN CREATE IT
        print,'Making flat field'
        qflat = where(strct.type eq 'FLT' and strct.objnum eq objnum,nflt)
        ldss_mkflat, strct, objnum
   endif
   flat = mrdfits(flatname)
   print, ' Flat field = ',flatname
   print


 ; DECIDE REGION TO ANALYZE
   ytrim = [1300,2500]
   ytrim = [1520,2720]
   yobj  = [200,1000]   ; trimed image


; FOR EACH OBJECT FRAME
  for i = 0, nobj-1 do begin

       ; STITCH OBJECT FRAME TOGETHER
         if keyword_set(nccd4) then $
         ldss_stitch4,strct, qobj[i], im, hdr=hdr else $
         ldss_stitch2,strct, qobj[i], im, hdr=hdr,/plot

    

       ; CREATE VARIENCE IMAGE:  var = S*gain + rn^2
       ;    not correct-- need to account for chip gains
         print,'Creating inverse varience image'
         var = ((im*0.7 + 3.7^2) > 0.)    
         q=where(var eq 0)
         ivar=1./var
         ivar[q] = 0



       ; FLAT FIELD
         im = im / flat
         ivar = ivar * flat^2
         atv, im[*,ytrim[0]:ytrim[1]],/align
 

       ; CREATE MASK IMAGE
         print,'Making mask image'
         ldss_mkmask, strct, objnum, mask
         im = im*mask
         ivar = ivar*mask
         q=where(not float(finite(im)))
         if (q[0] gt -1) then im[q] = 0
         if (q[0] gt -1) then ivar[q] = 0   ;  clean up NaNs
         imsky = im


       ; FLAG CR HITS  
         print,'Flagging CR hits'
         ldss_crflag, im, ivar,crflg;,/fixchip2
         im[crflg] = 0.
         ivar[crflg] = 0.
         atv,im,/align

       ; INTERPOLATE OVER BAD COLUMNS -- defined in LDSS3/ldss_badcol.dat
         ldss_fixcol,im, ivar


         im = im[*,ytrim[0]:ytrim[1]]
         ivar = ivar[*,ytrim[0]:ytrim[1]]
         imsky = im

       ; CREATE WAVELENGTH IMAGE
         print,'Creating 2D wavelength solution'
         arcfile = 'Arcs/arc_' + string(objnum) + '.fits'
         arcfile = strcompress(arcfile,/remove_all)
         a = file_search(arcfile, count = cnt)
         if (cnt eq 0) then begin
             ; IF ARC SOLUTION IMAGE DOESN'T EXIST, THEN CREATE IT
               ldss_arcsol,strct, objnum, wav ,ytrim = ytrim, /plot
         endif else wav = mrdfits(arcfile)       
         atv,wav


          ; SKY SUBTRACT   
            ldss_skysub, im, ivar, wav, imred, yobj=yobj, bksp = 1,/plot
            q=where(ivar eq 0)
            imred[q] = 0  ; re-zero masked pixels

            atv,imred
            

          ; WRITE TO FILE
            zr='000'
            if (strct[qobj[i]].ccdnum gt 9) then zr = '00'
            if (strct[qobj[i]].ccdnum gt 99) then zr = '0'
            if (strct[qobj[i]].ccdnum gt 999) then zr = ''
            ffile = 'Final/f_ccd'+zr +string(strct[qobj[i]].ccdnum)+'.fits'
            ffile = strcompress(ffile,/remove_all)


            mwrfits, 0, ffile, /create
            mwrfits, imred, ffile, hdr
            mwrfits, ivar, ffile
            mwrfits, wav, ffile
            mwrfits, imsky, ffile

    endfor

return
end
