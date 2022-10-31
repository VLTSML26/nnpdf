%Closure test: outputs of missing systematics and normal closuretest

Summary
-------

This report shows {@ fit @}'s results compared to its underlying law and to a normal closure test.

{@ with fitsummary @}

### {@ tablename @}

{@ closuretest_summary @}

{@ endwith @}

PDFs
----

{@ with scales @}
{@ with PDFscalespecs @}
### Plots at {@ scaletitle @}, xscale {@ Xscaletitle @}

{@ plot_pdfs @}

{@ endwith @}
{@ endwith @}

Replicas
--------

{@ with PDFscalespecs @}
### Xscale {@ Xscaletitle @}

{@ plot_pdfreplicas @}
{@ endwith @}