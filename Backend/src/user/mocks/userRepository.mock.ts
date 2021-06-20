export function MockUserModel(dto: any){
    this.data = dto;
    this.save = () => { return this.data};
  };