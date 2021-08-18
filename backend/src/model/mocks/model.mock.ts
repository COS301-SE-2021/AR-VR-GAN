import { join } from "path";
import { ReplaySubject } from "rxjs";
import { RequestProxy } from "../grpc.interface";

export const MockModelService= {

    handleCoords: jest.fn((dto) => {
        let sum = 0;

        for (let i = 0; i < dto.data.length; i++) {
            sum += dto.data[i]
        }
        return sum;
    }),

    runPython: jest.fn((request) => {
        var myPythonScriptPath = join(__dirname, '../../../../generativeModelFiles/modelGenerator.py');
        var myPythonModelPath = join(__dirname, '../../../../generativeModelFiles/defaultModels/Epochs-50.pt');

        const spawnSync = require("child_process").spawnSync;

        var commaSplitList = request.data.toString().split(',');
        var coord1 = parseFloat(commaSplitList[0]);
        var coord2 = parseFloat(commaSplitList[1]);
        var coord3 = parseFloat(commaSplitList[2]);

        var process = spawnSync('python',["-W ignore",myPythonScriptPath,"--coordinates",coord1,coord2,coord3,"--model",myPythonModelPath]);
        //console.log(process.stderr.toString())
        return process.stdout.toString()
    }),

    proxy: jest.fn((request) => {
        const subject = new ReplaySubject<RequestProxy>();
        subject.next({ vector: request.data });
        subject.complete();
        const stream = MockModelService.grpcService.generateImage(subject.asObservable());
        return stream.toPromise();
    })
  }
