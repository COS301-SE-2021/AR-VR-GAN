export class MockUserModel{
  constructor(public dto){}
  save = jest.fn().mockResolvedValue(this.dto)
  findOne(data):any {return null};
  MockUserModel(data){}
  userModel(data){}
};
