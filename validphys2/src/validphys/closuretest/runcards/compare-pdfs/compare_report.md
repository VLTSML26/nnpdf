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

### Plots at {@ scaletitle @}

{@ plot_pdfs @}

{@ endwith @}

Replicas
--------

{@ plot_pdfreplicas @}
