from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput
from merging_scenario_paper import experiment_merging_scenario



if __name__ == "__main__":
    # 设置输出格式
    graphviz = GraphvizOutput()
    graphviz.output_file = 'output.png'
    # 手动指定 dot 工具的路径
    graphviz.tool = "C:\\Users\mrayu\Downloads\windows_10_cmake_Release_Graphviz-12.2.1-win64\Graphviz-12.2.1-win64\\bin\dot"

    # 使用 PyCallGraph 捕获函数调用
    with PyCallGraph(output=graphviz):
        experiment_merging_scenario(True, True, True)