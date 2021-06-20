import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ProtoComponent } from './proto.component';

describe('ProtoComponent', () => {
  let component: ProtoComponent;
  let fixture: ComponentFixture<ProtoComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ProtoComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(ProtoComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
