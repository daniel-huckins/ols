package ols

import (
	"github.com/gonum/matrix/mat64"
)

// Model handles the regression
type Model struct {
	x            *mat64.Dense
	y            *mat64.Vector // one dimentional
	hasIntercept bool
}

// NewModel creates a new model with an intercept by default
func NewModel(x, y mat64.Matrix) *Model {
	return NewModelWithIntercept(x, y, true)
}

// NewModelWithIntercept creates a new OLS model with or without an intercept
// x or y can be nil, just needs to be updated before training
func NewModelWithIntercept(x, y mat64.Matrix, intercept bool) *Model {
	m := new(Model)
	m.hasIntercept = intercept
	m.SetX(x)
	m.SetY(y)

	return m
}

// SetX replaces the independent variables in the model
func (m *Model) SetX(data mat64.Matrix) {
	if data == nil {
		return
	}
	m.x = mat64.DenseCopyOf(data)
}

// SetY replaces the dependent model
// NOTE: only uses first column of the matrix
func (m *Model) SetY(y mat64.Matrix) {
	if y == nil {
		return
	}
	rows, _ := y.Dims()
	m.y = mat64.NewVector(rows, nil)
	for r := 0; r < rows; r++ {
		m.y.SetVec(r, y.At(r, 0))
	}
}

// Dims returns number of independent variabes and count of rows
func (m *Model) Dims() (int, int) {
	return m.x.Dims()
}

// Train - process the data and return the result
func (m *Model) Train() (res *mat64.Dense, err error) {
	// nrows, ncols := m.x.Dims()
	// inv := mat64.NewDense(nrows, ncols, nil)
	var inv *mat64.Dense
	xr, xc := m.x.Dims()
	xxr, xxc := m.x.T().Dims()
	log.Printf("x dims: (%d, %d)", xr, xc)
	log.Printf("x' dims: (%d, %d)", xxr, xxc)
	inv.Mul(m.x, m.x.T())
	inv.Inverse(inv)
	err = inv.Inverse(inv)
	if err != nil {
		return
	}
	res.Product(inv, m.x.T(), m.y)

	return
}
