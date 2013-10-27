;--------------------------------------------------------------------
;+
; NAME:
;      ldss_mkmask
; 
; PURPOSE:
;     Mask regions of zero flux (longslit bridge regions). 
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
Pro ldss_mkmask, strct, objnum, mask,PLOT=plot


   flatroot = 'Flats/flat_'
   flatname = flatroot + string(objnum) + '.fits'
   flatname = strcompress(flatname,/remove_all)


   mask = mrdfits(flatname)


 ; CREATE A MASK IMAGE FROM REDUCED FLATFIELD-- DEFINE UNUSED AND 
 ; LONGSLIT BRIDGE REGIONS  
   
  ; gdim = flat[*,1300:2500]   ; rough region of LDSS3 usable data-- slit specific!
   
   for jj = 0, 4063 do begin
       
       sig = djs_median(mask[jj,1500:2500])

       q=where(mask[jj,*] le sig*0.75)

       mask[jj,q] = 0
   endfor

   q=where(mask ne 0)
   mask[q] = 1

end
