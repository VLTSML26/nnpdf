// $Id
//
// NNPDF++ 2012
//
// Authors: Nathan Hartland,  n.p.hartland@ed.ac.uk
//          Stefano Carrazza, stefano.carrazza@mi.infn.it
//          Luigi Del Debbio, luigi.del.debbio@ed.ac.uk

#pragma once

#include "buildmaster_utils.h"

//Normalised distributions

class CMS_TTB_DIFF_13TEV_2016_2L_TPTNORMFilter: public CommonData
{
 public: CMS_TTB_DIFF_13TEV_2016_2L_TPTNORMFilter():
  CommonData("CMS_TTB_DIFF_13TEV_2016_2L_TPTNORM") { ReadData(); }

 private:
  void ReadData();
};

class CMS_TTB_DIFF_13TEV_2016_2L_TRAPNORMFilter: public CommonData
{
 public: CMS_TTB_DIFF_13TEV_2016_2L_TRAPNORMFilter():
  CommonData("CMS_TTB_DIFF_13TEV_2016_2L_TRAPNORM") { ReadData(); }

 private:
  void ReadData();
};

class CMS_TTB_DIFF_13TEV_2016_2L_TTMNORMFilter: public CommonData
{
 public: CMS_TTB_DIFF_13TEV_2016_2L_TTMNORMFilter():
  CommonData("CMS_TTB_DIFF_13TEV_2016_2L_TTMNORM") { ReadData(); }

 private:
  void ReadData();
};

class CMS_TTB_DIFF_13TEV_2016_2L_TTRAPNORMFilter: public CommonData
{
 public: CMS_TTB_DIFF_13TEV_2016_2L_TTRAPNORMFilter():
  CommonData("CMS_TTB_DIFF_13TEV_2016_2L_TTRAPNORM") { ReadData(); }

 private:
  void ReadData();
};
