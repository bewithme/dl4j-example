package cn.skymind.examples.utils;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import lombok.Data;

@Parameters(separators = "=", commandDescription = "运行参数")
@Data
public class AppRunOptions {

	 @Parameter(names = {"-entryClass"}, description = "入口类", required = true)
	 private String entryClass;
	 
	 @Parameter(names = "--help", help = true)
	 private boolean help;
	 
}
