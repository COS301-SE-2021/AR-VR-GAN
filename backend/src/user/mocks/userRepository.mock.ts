// export function MockUserModel(dto: any){
//     this.data = dto;
//     this.save = () => { return this.data};
//     this.findOne = null
//   };
  

export class MockUserModel{
  constructor(public dto){}
  save = jest.fn().mockResolvedValue(this.dto)
  findOne(data):any {return null};
  MockUserModel(data){}
  userModel(data){}
};

// export function MockUserModel(dto: any) {
//   this.data = dto;
//   this.save  = () => {
//     return this.data;
//   };
//   this.findOne  = () => {
//     return this.data;
//   };
// }