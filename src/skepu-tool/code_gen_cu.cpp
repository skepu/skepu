#include "code_gen_cl.h"

IndexCodeGen indexInitHelper_CU(UserFunction &uf)
{
	IndexCodeGen res;
	res.dim = 0;
	
	if (uf.indexed1D || uf.indexed2D || uf.indexed3D || uf.indexed4D)
	{
		res.mapFuncParam = "skepu_index";
		res.hasIndex = true;
	}
	
	if (uf.indexed1D)
	{
		res.dim = 1;
		res.indexInit = "skepu::Index1D skepu_index;\nskepu_index.i = skepu_base + skepu_i;";
	}
	else if (uf.indexed2D)
	{
		res.dim = 2;
		res.indexInit = "skepu::Index2D skepu_index;\nskepu_index.row = (skepu_base + skepu_i) / skepu_w2;\nskepu_index.col = (skepu_base + skepu_i) % skepu_w2;";
	}
	else if (uf.indexed3D)
	{
		res.dim = 3;
		res.indexInit = R"~~~(
			skepu::Index3D skepu_index;
			size_t skepu_cindex = skepu_base + skepu_i;
			skepu_index.i = skepu_cindex / (skepu_w2 * skepu_w3);
			skepu_cindex = skepu_cindex % (skepu_w2 * skepu_w3);
			skepu_index.j = skepu_cindex / (skepu_w3);
			skepu_index.k = skepu_cindex % (skepu_w3);
	)~~~";
	}
	else if (uf.indexed4D)
	{
		res.dim = 4;
		res.indexInit = R"~~~(
			skepu::Index4D skepu_index;
			size_t skepu_cindex = skepu_base + skepu_i;
			skepu_index.i = skepu_cindex / (skepu_w2 * skepu_w3 * skepu_w4);
			skepu_cindex = skepu_cindex % (skepu_w2 * skepu_w3 * skepu_w4);
			skepu_index.j = skepu_cindex / (skepu_w3 * skepu_w4);
			skepu_cindex = skepu_cindex % (skepu_w3 * skepu_w4);
			skepu_index.k = skepu_cindex / (skepu_w4);
			skepu_index.l = skepu_cindex % (skepu_w4);
	)~~~";
	}
	
	return res;
}






RandomAccessAndScalarsResult handleRandomAccessAndUniforms_CU(
	UserFunction &func,
	std::stringstream& SSMapFuncArgs,
	std::stringstream& SSKernelParamList,
	bool &first
)
{
	RandomAccessAndScalarsResult res;
	std::stringstream SSProxiesInit, SSProxiesUpdate;
	
	// Random-access data
	for (UserFunction::RandomAccessParam& param : func.anyContainerParams)
	{
		if (!first) { SSMapFuncArgs << ", "; }
		SSKernelParamList << param.unqualifiedFullTypeName << " " << param.name << ", ";
		SSMapFuncArgs << param.name;
		SSProxiesInit << param.resolvedTypeName << "* skepu_" << param.name << "_base = " << param.name << ".data;\n";
		first = false;
		switch (param.containerType)
		{
			case ContainerType::MatRow:
				SSProxiesUpdate << param.name << ".data = skepu_" << param.name << "_base + skepu_i * " << param.name << ".cols;\n";
				break;
			
			case ContainerType::MatCol:
				SSProxiesUpdate << param.name << ".data = skepu_" << param.name << "_base + skepu_i;\n";
				break;
		}
	}

	// Scalar input data
	for (UserFunction::Param& param : func.anyScalarParams)
	{
		if (!first) { SSMapFuncArgs << ", "; }
		SSKernelParamList << param.resolvedTypeName << " " << param.name << ", ";
		SSMapFuncArgs << param.name;
		first = false;
	}
	
	res.proxyInitializer = SSProxiesInit.str();
	res.proxyInitializerInner = SSProxiesUpdate.str();
	return res;
}

void handleRandomParam_CU(
	UserFunction &func,
	std::stringstream& SSMapFuncArgs,
	std::stringstream& SSKernelParamList,
	bool &first
)
{
	if (UserFunction::RandomParam *param = func.randomParam)
	{
		if (!first) { SSMapFuncArgs << ", "; }
		first = false;
		SSMapFuncArgs << param->name << "[skepu_global_prng_id]";
		SSKernelParamList << "skepu::Random<" << param->randomCount << ">* " << param->name << ", ";
	}
	else
	{
		SSKernelParamList << "skepu::PRNG::Placeholder,";
	}
}



std::string handleOutputs_CU(UserFunction &func, std::stringstream &SSKernelParamList, bool strided, std::string index)
{
	std::stringstream SSOutputBindings;
	if (func.multipleReturnTypes.size() == 0)
	{
		std::stringstream strideinc;
		if (strided) strideinc << " * skepu_strides[0]";
		SSKernelParamList << func.resolvedReturnTypeName << "* skepu_output, ";
		SSOutputBindings << "skepu_output[" << index << strideinc.str() << "] = skepu_res;";
	}
	else
	{
		size_t outCtr = 0;
		for (std::string& outputType : func.multipleReturnTypes)
		{
			std::stringstream strideinc;
			if (strided) strideinc << " * skepu_strides[" << outCtr << "]";
			SSKernelParamList << outputType << "* skepu_output_" << outCtr << ", ";
			SSOutputBindings << "skepu_output_" << outCtr << "[" << index << strideinc.str() << "] = skepu_res.e" << outCtr << ";\n";
			outCtr++;
		}
	}
	
	return SSOutputBindings.str();
}





std::string generateCUDAMultipleReturn(UserFunction &UF)
{
	if (UF.multipleReturnTypes.size() > 0)
	{
		std::stringstream SSmultiReturnType, SSmultiReturnTypeDef, SSmultiReturnMakeStruct, SSmultiReturnMakeParams;
		SSmultiReturnType << "skepu_multiple";
		size_t ctr = 0;
		for (std::string &type : UF.multipleReturnTypes)
		{
			std::string divider = (ctr != UF.multipleReturnTypes.size() - 1) ? ", " : "";
			SSmultiReturnType << "_" << transformToCXXIdentifier(type);
			SSmultiReturnTypeDef << type << " e" << ctr << ";\n";
			SSmultiReturnMakeParams << type << " arg" << ctr << divider;
			SSmultiReturnMakeStruct << "arg" << ctr << divider;
			ctr++;
		}

		std::string codeTemplate = R"~~~(
			struct {{SKEPU_MULTIPLE_RETURN_TYPE}}
			{
				{{SKEPU_MULTIPLE_RETURN_TYPE_DEF}}

				static __device__ {{SKEPU_MULTIPLE_RETURN_TYPE}} make({{SKEPU_MULTIPLE_RETURN_MAKE_PARAMS}})
				{
					{{SKEPU_MULTIPLE_RETURN_TYPE}} retval = { {{SKEPU_MULTIPLE_RETURN_MAKE_STRUCT}} };
					return retval;
				}
			};
		)~~~";

		replaceTextInString(codeTemplate, "{{SKEPU_MULTIPLE_RETURN_TYPE}}", SSmultiReturnType.str());
		replaceTextInString(codeTemplate, "{{SKEPU_MULTIPLE_RETURN_TYPE_DEF}}", SSmultiReturnTypeDef.str());
		replaceTextInString(codeTemplate, "{{SKEPU_MULTIPLE_RETURN_MAKE_PARAMS}}", SSmultiReturnMakeParams.str());
		replaceTextInString(codeTemplate, "{{SKEPU_MULTIPLE_RETURN_MAKE_STRUCT}}", SSmultiReturnMakeStruct.str());

		return codeTemplate;
	}
	return "";
}