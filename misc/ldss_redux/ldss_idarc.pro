;--------------------------------------------------------------------
;+
; NAME:
;      ldss_idarc
; 
; PURPOSE:
;     Create a rough arc solution based on combined arc images.
;     write to local file.  will use this as the starting point 
;     for identifying a 2D arc solution for single arc frames taken 
;     at same position as science data.
;
; CALLING SEQUENCE:
;     ldss_idarc, strct
;
; INPUTS:
;     strct  -- LDSS structure, 
;     Requires LDSS3 HeArNe arc line list (ldss_henear.dat) 
;
; OPTIONAL INPUTS
;
;     hand_lambda  --  Array of wavelengths corresponding to
;                     identified arc lamp lines in arc frame.
;     hand_pix     -- Corresponding pixel values for arc lines
;
;
; OUTPUTS:
;     writes rough arc solution to Arc/arcsol.fits
;
; COMMENTS
;
;   CAUTION!  This has only been tested for the VPH blue grism and
;   red slit combination.  The file 'ldss_henear.dat' currently 
;   contains only arc lines between 4500 - 6800AA.  For other wavelength
;   regimes will need to modify this file.
;
;
; MODIFICATION HISTORY:
;     mg 3/05
;- 
;--------------------------------------------------------------------
Pro ldss_idarc, strct, hand_lambda=hand_lambda, hand_pix=hand_pix, PLOT=plot,$
                nccd4=nccd4

   !p.multi=0
   loadct,12

 ; LDSS3 HeAR ARC LIST
   dodgy_path = getenv('DODGY_DIR')
   arclist = dodgy_path +'/pro/LDSS3/ldss_henear.dat'
   readcol,arclist,linelist 


 ; INITIALIZE ARC SOLUTION WITH A FEW LINES -- set for VPH blue + redslit;
;   if NOT keyword_set(hand_lambda) then $
;       hand_lambda = [4471.47, 5015.675, 5944.834, 6402.246]
;   if NOT keyword_set(hand_pix)    then $
;       hand_pix    = [342,     1172,     2507,      3146]
   if NOT keyword_set(hand_lambda) then $
       hand_lambda = [4471.47, 5015.675, 5944.834, 6402.246];,6717.0]
   if NOT keyword_set(hand_pix)    then $
       hand_pix    = [364,     1193,     2525,      3163];, 3610]


 ; IF COMBINED ARC EXISTS, READ AND SKIP BELOW
   arcname = 'Arcs/arcid.fits'
   a = file_search(arcname, count = cnt)
   if (cnt eq 1) then begin
       print, 'READING EXISTING COMBINED ARC FILE'
       comb = mrdfits(arcname)
       goto,skip
   endif

 ; AVOID OVERKILL -- combine at most 9 arc frames
   qarc = where(strct.type eq 'ARC',narc)   
   if (narc ge 9) then narc = 9
   print,'Combining '+string(narc)+' arc frames'


   allarc = fltarr(4064, 4064,narc)
   for i=0,narc-1 do begin

       indx = qarc[i]
       if keyword_set(nccd4) then ldss_stitch4,strct,indx,imfinal $
         else  ldss_stitch2,strct,indx,imfinal 
       allarc[*,*,i] = imfinal

   endfor

 ; CREATE ARC FRAME from median of many arcs 
   if(narc gt 1) then comb = djs_median(allarc,3)
   if(narc eq 1) then comb = imfinal
   mwrfits,comb,arcname,/create

 ; USE ONLY A SMALL PORTION OF ARC IN Y-AXIS 
   skip:a=1 
   arc = comb[*,1800:2400]
;   if keyword_set(plot) then atv,arc

  
 ; PLOT ARC FRAME`
   carc = djs_median(arc,2)
   xpix = findgen(n_elements(carc))
   plot,xpix,carc,xstyle=1,yrange=[-50,1e3],ystyle=1

   plot,xpix,carc,xstyle=1,yrange=[-50,1e3],ystyle=1,xrange=[3500,4000]



 ; IDENTIFY ARC LINES-- do this crudely by hand for three lines
 ;  hand_lambda = [4471.47, 5015.675, 5944.834, 6402.246]
 ;  hand_pix    = [342,     1172,     2507,      3146]
   xyouts,hand_pix,-100,'*',color=180,alignment=0.5


 ; FIT TO HAND PICKED LINES, PREDICT XPIX FOR LINELIST
   afit = poly_fit(hand_lambda, hand_pix,3)
   lpix = afit(0) + afit(1) * linelist + afit(2) * linelist^2 + afit(3) * linelist^3
   xyouts,lpix,0,'|',color=140,alignment=0.5

stop  

 ; FIT GAUSSIAN TO EACH LINE, DETERMINE CENTER-- this will be used
 ; as beginning positions for more accurate 2D fit.
  !p.multi=[0,5,5]
   linelist_pix = lpix
   openw,1,'Arcs/arcid.dat'
   for i=0,n_elements(linelist) -1 do begin
      
      ; CENTRIOD EACH LINE BY FITTING A GAUSSIAN
        qfit = where(xpix ge lpix[i]-8 and xpix le lpix[i]+8)
        g=gaussfit(xpix[qfit],carc[qfit],a)
        linelist_pix[i] = a(1)
        
        if keyword_set(plot) then begin
            plot,xpix[qfit],carc[qfit]
            oplot,xpix[qfit],g,color=100
            oplot,[a(1),a(1)],[0,1e6]
            stop
        endif

      ; WRITE LINE/XPIX TO FILE 
        printf,1,linelist_pix[i],linelist[i]
        print,linelist_pix[i],linelist[i]

  endfor

  close,1
  xyouts,linelist_pix,0,'|',color=100,alignment=0.5


  

end
